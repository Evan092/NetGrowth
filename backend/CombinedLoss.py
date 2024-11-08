import math
import scipy
import torch
import torch.nn as nn
import torchvision
import torchvision.ops as ops
import Constants
from GIoULoss import GIoULoss
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_single_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (Tensor): Tensor of shape [4], format (x1, y1, x2, y2)
        box2 (Tensor): Tensor of shape [4], format (x1, y1, x2, y2)

    Returns:
        Tensor: IoU value as a scalar tensor.
    """
    # Ensure the boxes are floats for division
    box1 = box1.float()
    box2 = box2.float()

    # Intersection coordinates
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])

    # Compute the width and height of the intersection rectangle
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)

    # Compute the area of intersection
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of union
    union_area = area_box1 + area_box2 - inter_area

    # Compute IoU, adding a small epsilon to avoid division by zero
    iou = inter_area / (union_area + 1e-6)

    return iou

def filter_confidences(pred_boxes, confidences, threshold=0.5):
    """
    Filter predicted boxes and their confidences based on a confidence threshold.
    
    Parameters:
    - pred_boxes: Tensor of predicted boxes [N, 4].
    - confidences: Tensor of confidence scores [N, 1].
    - threshold: Float, the minimum confidence score to keep a box.
    
    Returns:
    - filtered_boxes: Tensor of filtered boxes.
    - filtered_confidences: Tensor of filtered confidence scores, retaining the original shape [M, 1].
    """
    mask = confidences.squeeze(1) >= threshold  # Use squeeze(1) to safely remove the singleton dimension for comparison

    # Use the mask to filter out boxes and confidences
    filtered_boxes = pred_boxes[mask]  # Apply the mask to the boxes
    filtered_confidences = confidences[mask]  # Apply the mask to the confidences

    # Reshape if necessary to ensure correct dimensions
    if filtered_boxes.dim() == 1:  # Only one box passed the filter
        filtered_boxes = filtered_boxes.unsqueeze(0)
    if filtered_confidences.dim() == 1:  # Only one confidence score passed the filter
        filtered_confidences = filtered_confidences.unsqueeze(0)

    return filtered_boxes, filtered_confidences

def calculate_iou(box1, box2):
    """Compute IoU between two sets of boxes."""
    # Get coordinates
    x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0])
    y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1])
    x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2])
    y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3])
    
    # Compute intersection area
    inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    
    # Compute areas of individual boxes
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Compute union area
    union_area = box1_area.unsqueeze(1) + box2_area - inter_area
    
    # Avoid division by zero by adding a small epsilon value
    iou = inter_area / (union_area + 1e-6)
    
    return iou

def calculate_target_conf(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Calculate target_conf for each predicted box.
    
    Args:
        pred_boxes (Tensor): Tensor of shape (N, 4) where N is the number of predicted boxes.
        gt_boxes (Tensor): Tensor of shape (M, 4) where M is the number of ground truth boxes.
        iou_threshold (float): IoU threshold for positive detection.

    Returns:
        target_conf (Tensor): Tensor of shape (N,) with values 0 or 1.
    """
    # Calculate IoUs between all predicted boxes and ground truth boxes
    ious = calculate_iou(pred_boxes, gt_boxes)

    # Determine maximum IoU for each predicted box
    max_ious, _ = ious.max(dim=1)

    # Assign confidence based on IoU threshold
    target_conf = (max_ious >= iou_threshold).float()

    return target_conf


def apply_nms(pred_boxes, confidences, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to reduce overlapping boxes and ensure the confidence tensor's shape is maintained as [N, 1].

    Parameters:
    - pred_boxes: Tensor of filtered predicted boxes [N, 4].
    - confidences: Tensor of filtered confidence scores [N, 1].
    - iou_threshold: Float, the IoU threshold for filtering overlaps.

    Returns:
    - kept_boxes: Tensor of boxes kept after NMS.
    - kept_confidences: Tensor of confidence scores of kept boxes, shape [K, 1] where K is the number of boxes kept.
    """
    # Apply NMS using the confidences squeezed to one dimension for proper NMS functionality
    keep_indices = torchvision.ops.nms(pred_boxes, confidences.squeeze(1), iou_threshold)

    # Index the results with keep_indices. The result for boxes is straightforward.
    kept_boxes = pred_boxes[keep_indices]

    # For confidences, ensure the output shape is [K, 1] by using keep_indices
    kept_confidences = confidences[keep_indices]

    return kept_boxes, kept_confidences


def assign_boxes(pred_boxes, target_boxes):
    """
    Assign each ground-truth box to the closest predicted box based on center distance.
    
    Parameters:
        pred_boxes: Tensor of shape [batch_size, num_anchors, grid_h, grid_w, 4]
        target_boxes: Tensor of shape [batch_size, max_gt_boxes, 4]
    
    Returns:
        assigned_mask: Tensor of shape [batch_size, num_anchors, grid_h, grid_w]
                    indicating which predicted boxes correspond to ground-truth boxes.
    """
    batch_size, num_anchors, grid_h, grid_w, _ = pred_boxes.shape
    max_gt_boxes = target_boxes.shape[1]

    # Calculate the centers of predicted and ground-truth boxes
    pred_centers = (pred_boxes[..., :2] + pred_boxes[..., 2:4]) / 2  # (x1, y1) -> (center_x, center_y)
    target_centers = (target_boxes[..., :2] + target_boxes[..., 2:4]) / 2  # Same for ground-truth

    # Initialize an assignment mask (all False initially)
    assigned_mask = torch.zeros((batch_size, num_anchors, grid_h, grid_w), dtype=torch.bool, device=pred_boxes.device)

    # Loop through each image in the batch
    for b in range(batch_size):
        for t in range(max_gt_boxes):
            # Get the ground-truth box center for this box
            gt_center = target_centers[b, t]

            # Calculate the Euclidean distance between this GT box and all predicted boxes
            distances = torch.sqrt(((pred_centers[b] - gt_center) ** 2).sum(dim=-1))  # [num_anchors, grid_h, grid_w]

            # Find the anchor/grid cell with the smallest distance to the GT box
            min_distance_idx = torch.argmin(distances)

            # Mark this prediction as assigned to the ground-truth box
            assigned_mask[b].view(-1)[min_distance_idx] = True

    return assigned_mask



def diou_batch(pred_boxes, target_boxes):
    """Vectorized DIoU calculation between predictions and ground-truth boxes."""
    # Expand dimensions for broadcasting: [batch_size, N, 1, 4], [batch_size, 1, M, 4]
    pred_boxes = pred_boxes.unsqueeze(2)  # [batch_size, N, 1, 4]
    target_boxes = target_boxes.unsqueeze(1)  # [batch_size, 1, M, 4]

    # Intersection coordinates
    inter_x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
    inter_y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
    inter_x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
    inter_y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    area2 = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    union_area = area1 + area2 - inter_area

    iou = inter_area / (union_area + 1e-6)

    # Center distances
    pred_center_x = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2
    pred_center_y = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2
    target_center_x = (target_boxes[..., 0] + target_boxes[..., 2]) / 2
    target_center_y = (target_boxes[..., 1] + target_boxes[..., 3]) / 2

    center_dist = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2

    # Enclosing box diagonal distance
    enclose_x1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
    enclose_y1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
    enclose_x2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
    enclose_y2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
    enclose_dist = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    return iou - (center_dist / (enclose_dist + 1e-6))  # DIoU formula


import torch

def filter_and_trim_boxes2(pred_boxes, target_boxes, max_boxes=None, iou_threshold=0.5):
    batch_size, num_anchors, grid_h, grid_w, num_outputs = pred_boxes.shape

    # Reshape predictions to [batch_size, N, 5]
    pred_boxes = pred_boxes.view(batch_size, -1, 5)  # N = num_anchors * grid_h * grid_w
    pred_coords = pred_boxes[..., :4]  # [batch_size, N, 4]
    pred_confidences = pred_boxes[..., 4:5]  # [batch_size, N, 1]

    valid_mask = (target_boxes.sum(dim=-1) > 0)  # [batch_size, num_boxes]

    filtered_target_boxes = []
    filtered_pred_boxes = []
    filtered_confidences = []

    for i in range(batch_size):
        valid_boxes = target_boxes[i][valid_mask[i]]  # [num_valid_boxes, 4]
        if valid_boxes.numel() == 0:
            continue  # Skip if no valid boxes

        pred_coords_i = pred_coords[i]  # [N, 4]
        pred_confidences_i = pred_confidences[i]  # [N, 1]

        # Compute DIoU between all predicted and valid ground-truth boxes
        dious = diou(pred_coords_i, valid_boxes)  # [N_pred, N_gt]

        # Convert DIoU to cost matrix
        cost_matrix = 1 - dious  # Lower cost is better

        # Apply Hungarian algorithm on GPU
        matched_pred_indices, matched_gt_indices = hungarian(cost_matrix)

        # Filter matches based on IoU threshold
        #valid_matches = dious[matched_pred_indices, matched_gt_indices].bool() # iou_threshold
        #matched_pred_indices = matched_pred_indices[valid_matches]
        #matched_gt_indices = matched_gt_indices[valid_matches]

        # Collect assigned predicted boxes and their confidences
        filtered_pred_coords = pred_coords_i[matched_pred_indices]
        filtered_confidences_flat = pred_confidences_i[matched_pred_indices]

        filtered_target_boxes.append(valid_boxes[matched_gt_indices])
        filtered_pred_boxes.append(filtered_pred_coords)
        filtered_confidences.append(filtered_confidences_flat)

    # Concatenate results across the batch
    if filtered_target_boxes:
        target_boxes_flat = torch.cat(filtered_target_boxes, dim=0)
        pred_boxes_flat = torch.cat(filtered_pred_boxes, dim=0)
        confidences_flat = torch.cat(filtered_confidences, dim=0)
    else:
        target_boxes_flat = torch.empty(0, 4, device=pred_boxes.device)
        pred_boxes_flat = torch.empty(0, 4, device=pred_boxes.device)
        confidences_flat = torch.empty(0, 1, device=pred_boxes.device)

    return target_boxes_flat, pred_boxes_flat, confidences_flat

def hungarian(cost_matrix):
    """
    Implements the Hungarian algorithm using PyTorch tensors.
    Args:
        cost_matrix (torch.Tensor): [num_preds, num_gts]
    Returns:
        matched_pred_indices (torch.Tensor): Indices of matched predictions
        matched_gt_indices (torch.Tensor): Indices of matched ground truths
    """
    num_preds, num_gts = cost_matrix.shape
    cost_matrix = cost_matrix.clone()

    # Initialize
    u = torch.zeros(num_preds, device=cost_matrix.device)
    v = torch.zeros(num_gts, device=cost_matrix.device)
    ind_pred = torch.full((num_preds,), -1, dtype=torch.long, device=cost_matrix.device)
    ind_gt = torch.full((num_gts,), -1, dtype=torch.long, device=cost_matrix.device)

    # For simplicity, we use a basic implementation suitable for small matrices.
    # For larger matrices, consider using optimized libraries or implementations.
    for _ in range(num_preds):
        # Find the minimal uncovered element
        min_value = cost_matrix.min()
        if min_value == float('inf'):
            break

        pred_idx, gt_idx = (cost_matrix == min_value).nonzero(as_tuple=True)
        pred_idx = pred_idx[0]
        gt_idx = gt_idx[0]

        # Assign and cover the row and column
        ind_pred[pred_idx] = gt_idx
        ind_gt[gt_idx] = pred_idx
        cost_matrix[pred_idx, :] = float('inf')
        cost_matrix[:, gt_idx] = float('inf')

    # Filter out unassigned predictions and ground truths
    matched_pred_indices = (ind_pred != -1).nonzero(as_tuple=True)[0]
    matched_gt_indices = ind_pred[matched_pred_indices]

    return matched_pred_indices, matched_gt_indices


def filter_and_trim_boxes(pred_boxes, target_boxes, max_boxes=Constants.max_boxes, iou_threshold=0.5):
    batch_size, _, _ = pred_boxes.shape

    # Reshape predictions to [batch_size, N, 5]
    #pred_boxes[0] = pred_boxes[0].view(batch_size, -1, 5)  # N = num_anchors * grid_h * grid_w
    #pred_boxes[1] = pred_boxes[1].view(batch_size, -1, 5)  # N = num_anchors * grid_h * grid_w
    #pred_boxes[2] = pred_boxes[2].view(batch_size, -1, 5)  # N = num_anchors * grid_h * grid_w

    #pred_boxes = torch.cat(pred_boxes, dim=1)

    pred_coords = pred_boxes[..., :4]  # [batch_size, N, 4]
    pred_confidences = pred_boxes[..., 4]  # [batch_size, N]

    #pred_coords = yolo_to_corners_batches(pred_coords)

    pred_coords = torch.clamp(pred_coords, -10000, 10000)

    valid_mask = (target_boxes.sum(dim=-1) > 0)

    filtered_target_boxes = []
    filtered_pred_boxes = []
    filtered_confidences = []

    for i in range(batch_size):
        valid_boxes = target_boxes[i][valid_mask[i]]
        if valid_boxes.numel() == 0:
            continue  # Skip if no valid boxes

        # Compute DIoU between all predicted and valid ground-truth boxes
        dious = diou(pred_coords[i], valid_boxes)  # [N, num_valid]

        # Allow multiple matches based on threshold
        matched_indices = (dious > iou_threshold).nonzero(as_tuple=True)

        # If no matches exceed the threshold, fallback to the best DIoU
        if matched_indices[0].numel() == 0 or True:
            
            # Convert 'dious' to a NumPy array
            dious_np = dious.detach().cpu().numpy()

            # Compute the cost matrix
            cost_matrix = 1 - dious_np  # Use '1 - dious_np' if DIoU values are between 0 and 1

            # Apply the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Convert the indices back to tensors if needed
            matched_pred_indices = torch.from_numpy(row_ind).to(dious.device)
            matched_gt_indices = torch.from_numpy(col_ind).to(dious.device)

            #matched_pred_indices, matched_gt_indices = apply_greedy_matching(dious)


            # Now, 'matched_pred_indices' and 'matched_gt_indices' are your matched indices without repeats
            matched_indices = (matched_pred_indices, matched_gt_indices)

        # Collect assigned predicted boxes and their confidences
        assigned_mask = torch.zeros(pred_coords.shape[1], dtype=torch.bool, device=pred_coords.device)
        assigned_mask[matched_indices[0]] = True

        filtered_pred_coords = pred_coords[i][assigned_mask]
        filtered_confidences_flat = pred_confidences[i][assigned_mask].unsqueeze(-1)

        filtered_target_boxes.append(valid_boxes[matched_indices[1]])
        filtered_pred_boxes.append(filtered_pred_coords)
        filtered_confidences.append(filtered_confidences_flat)

    # Concatenate results across the batch
    target_boxes_flat = torch.cat(filtered_target_boxes, dim=0)
    pred_boxes_flat = torch.cat(filtered_pred_boxes, dim=0)
    confidences_flat = torch.cat(filtered_confidences, dim=0)

    return target_boxes_flat, pred_boxes_flat, confidences_flat




def filter_and_trim_boxes4(pred_boxes, target_boxes, max_boxes=Constants.max_boxes, iou_threshold=0.5):
    batch_size, num_anchors, grid_h, grid_w, num_outputs = pred_boxes.shape

    # Reshape predictions to [batch_size, N, 5]
    pred_boxes = pred_boxes.view(batch_size, -1, 5)
    pred_coords = pred_boxes[..., :4]
    pred_confidences = pred_boxes[..., 4]

    valid_mask = (target_boxes.sum(dim=-1) > 0)

    filtered_target_boxes = []
    filtered_pred_boxes = []
    filtered_confidences = []

    for i in range(batch_size):
        valid_boxes = target_boxes[i][valid_mask[i]]
        if valid_boxes.numel() == 0:
            continue  # Skip if no valid boxes

        # Compute DIoU between all predicted and valid ground-truth boxes
        dious = diou(pred_coords[i], valid_boxes)  # [N, num_valid]

        # Allow multiple matches based on threshold
        matched_indices = (dious > iou_threshold).nonzero(as_tuple=True)

        # If no matches exceed the threshold, fallback to the best DIoU
        if matched_indices[0].numel() == 0:
            best_matches = torch.argmax(dious, dim=0)
            matched_indices = (best_matches, torch.arange(valid_boxes.shape[0]))

        # Collect assigned predicted boxes and their confidences
        assigned_mask = torch.zeros(pred_coords.shape[1], dtype=torch.bool, device=pred_coords.device)
        assigned_mask[matched_indices[0]] = True

        filtered_pred_coords = pred_coords[i][assigned_mask]
        filtered_confidences_flat = pred_confidences[i][assigned_mask].unsqueeze(-1)

        filtered_target_boxes.append(valid_boxes)
        filtered_pred_boxes.append(filtered_pred_coords)
        filtered_confidences.append(filtered_confidences_flat)

    # Concatenate results across the batch
    target_boxes_flat = torch.cat(filtered_target_boxes, dim=0)
    pred_boxes_flat = torch.cat(filtered_pred_boxes, dim=0)
    confidences_flat = torch.cat(filtered_confidences, dim=0)

    return target_boxes_flat, pred_boxes_flat, confidences_flat

def filter_and_trim_boxes3(pred_boxes, target_boxes, max_boxes=Constants.max_boxes, iou_threshold=0.5):
    batch_size, num_anchors, grid_h, grid_w, num_outputs = pred_boxes.shape

    # Reshape predictions to [batch_size, num_anchors * grid_h * grid_w, 5]
    pred_boxes = pred_boxes.view(batch_size, -1, 5)
    pred_coords = pred_boxes[..., :4]  # [batch_size, N, 4]
    pred_confidences = pred_boxes[..., 4]  # [batch_size, N]

    valid_mask = (target_boxes.sum(dim=-1) > 0)  # Mask to filter valid ground-truth boxes

    filtered_target_boxes = []
    filtered_pred_boxes = []
    filtered_confidences = []

    for i in range(batch_size):
        valid_boxes = target_boxes[i][valid_mask[i]]  # [num_valid, 4]
        num_valid = valid_boxes.shape[0]

        assigned_mask = torch.zeros(pred_coords.shape[1], dtype=torch.bool, device=pred_boxes.device)

        for t in range(num_valid):
            gt_box = valid_boxes[t].unsqueeze(0)  # [1, 4]
            dious = diou(pred_coords[i], gt_box)  # [N]

            # Allow multiple anchors to match if DIoU exceeds the threshold
            matched_indices = torch.where(dious > iou_threshold)[0]

            # If no valid match is found, fall back to the best DIoU
            if len(matched_indices) == 0:
                best_match = torch.argmax(dious)
                matched_indices = torch.tensor([best_match], dtype=torch.long)

            # Mark these predictions as assigned
            assigned_mask[matched_indices] = True

        # Collect assigned predicted boxes and their confidences
        filtered_pred_coords = pred_coords[i][assigned_mask]  # Tensor [num_assigned, 4]
        filtered_confidences_flat = pred_confidences[i][assigned_mask].unsqueeze(-1)  # Tensor [num_assigned, 1]

        # Append the filtered results for this image
        filtered_target_boxes.append(valid_boxes)
        filtered_pred_boxes.append(filtered_pred_coords)
        filtered_confidences.append(filtered_confidences_flat)

    # Concatenate results across all images in the batch
    target_boxes_flat = torch.cat(filtered_target_boxes, dim=0)  # [total_valid, 4]
    pred_boxes_flat = torch.cat(filtered_pred_boxes, dim=0)      # [total_valid, 4]
    confidences_flat = torch.cat(filtered_confidences, dim=0)    # [total_valid, 1]

    return target_boxes_flat, pred_boxes_flat, confidences_flat

def filter_and_trim_boxes2(pred_boxes, target_boxes, max_boxes=Constants.max_boxes):
    batch_size, num_anchors, grid_h, grid_w, num_outputs = pred_boxes.shape

    # Reshape pred_boxes to [batch_size, grid_size * grid_size * num_boxes, 5]
    pred_boxes = pred_boxes.view(batch_size, -1, 5)

    # Split pred_boxes into coordinates and confidences
    pred_coords = pred_boxes[..., :4]  # [batch_size, N, 4]
    pred_confidences = pred_boxes[..., 4]  # [batch_size, N]

    # Step 1: Create a valid mask for target_boxes to filter out [0, 0, 0, 0] boxes
    valid_mask = (target_boxes.sum(dim=-1) > 0)  # Shape: [batch_size, max_boxes]

    # Step 2: For each image in the batch, filter out the padded boxes and match the predictions
    filtered_target_boxes = []
    filtered_pred_boxes = []
    filtered_confidences = []

                # Calculate the centers of predicted and ground-truth boxes
    pred_centers = (pred_boxes[..., :2] + pred_boxes[..., 2:4]) / 2  # (x1, y1) -> (center_x, center_y)
    target_centers = (target_boxes[..., :2] + target_boxes[..., 2:4]) / 2  # Same for ground-truth

        # Initialize an assignment mask (all False initially)

    for i in range(batch_size):
        # Get valid target boxes for the current image
        valid_boxes = target_boxes[i][valid_mask[i]]  # Shape: [num_valid, 4]

        # Number of valid target boxes
        num_valid = valid_boxes.shape[0]

        assigned_mask = torch.zeros((num_anchors * grid_h * grid_w,), dtype=torch.bool, device=pred_boxes.device)

        for t in range(num_valid):
            # Get the ground-truth box center for this box
            gt_center = target_centers[i, t]

            # Calculate the Euclidean distance between this GT box and all predicted boxes
            distances = torch.sqrt(((pred_centers[i] - gt_center) ** 2).sum(dim=-1))  # [num_anchors, grid_h, grid_w]

            # Find the anchor/grid cell with the smallest distance to the GT box
            min_distance_idx = torch.argmin(distances)

            # Mark this prediction as assigned to the ground-truth box
            assigned_mask[min_distance_idx] = True

        # Sort predicted boxes by confidence (descending)
        #_, topk_indices = torch.topk(pred_confidences[i], num_valid, largest=True)

        # Select the top-k most confident predicted boxes and their confidences
        #filtered_pred_coords = pred_coords[i][topk_indices]  # Shape: [num_valid, 4]
        #filtered_confidences_flat = pred_confidences[i][topk_indices].unsqueeze(-1)  # [num_valid, 1]

        #refiltered_pred_boxes, refiltered_confidences = filter_confidences(filtered_pred_coords, filtered_confidences_flat)
        #nmsfiltered_pred_boxes, nmsfiltered_confidences = apply_nms(refiltered_pred_boxes, refiltered_confidences)
        # Collect assigned predicted boxes and their confidences
        filtered_pred_coords = pred_coords [i][assigned_mask]  # Tensor [num_assigned, 4]
        filtered_confidences_flat = pred_confidences[i][assigned_mask].unsqueeze(-1)  # Tensor [num_assigned, 1]

        # Append the filtered results for this image
        filtered_target_boxes.append(valid_boxes)
        filtered_pred_boxes.append(filtered_pred_coords)
        filtered_confidences.append(filtered_confidences_flat)





    # Step 3: Concatenate results from all images in the batch
    target_boxes_flat = torch.cat(filtered_target_boxes, dim=0)  # Shape: [total_valid, 4]
    pred_boxes_flat = torch.cat(filtered_pred_boxes, dim=0)      # Shape: [total_valid, 4]
    confidences_flat = torch.cat(filtered_confidences, dim=0)    # Shape: [total_valid, 1]


    return target_boxes_flat, pred_boxes_flat, confidences_flat

def diou2(box1, box2):
    """Compute DIoU between two sets of boxes."""
    # Intersection coordinates
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    # Intersection and union areas
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = area1 + area2 - inter_area

    # IoU calculation
    iou = inter_area / union_area

    # Compute the center distance
    center_x1 = (box1[..., 0] + box1[..., 2]) / 2
    center_y1 = (box1[..., 1] + box1[..., 3]) / 2
    center_x2 = (box2[..., 0] + box2[..., 2]) / 2
    center_y2 = (box2[..., 1] + box2[..., 3]) / 2

    dist = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

    # Compute the diagonal of the smallest enclosing box
    enclose_x1 = torch.min(box1[..., 0], box2[..., 0])
    enclose_y1 = torch.min(box1[..., 1], box2[..., 1])
    enclose_x2 = torch.max(box1[..., 2], box2[..., 2])
    enclose_y2 = torch.max(box1[..., 3], box2[..., 3])
    diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    return iou - (dist / diag)  # DIoU formula

def diou(boxes1, boxes2):
    """Compute DIoU between two sets of boxes in a vectorized manner."""
    # Ensure boxes1 and boxes2 have the same shape for broadcasting
    #boxes1 = boxes1.unsqueeze(0)  # [batch_size, N, 1, 4]
    #boxes2 = boxes2.unsqueeze(0)  # [1, num_valid, 4]

        # Expand dimensions to enable pairwise operations
    boxes1_exp = boxes1.unsqueeze(1)  # [num_valid1, 1, 4]
    boxes2_exp = boxes2.unsqueeze(0)  # [1, num_valid2, 4]

    # Calculate Intersection Coordinates
    inter_x1 = torch.max(boxes1_exp[..., 0], boxes2_exp[..., 0])
    inter_y1 = torch.max(boxes1_exp[..., 1], boxes2_exp[..., 1])
    inter_x2 = torch.min(boxes1_exp[..., 2], boxes2_exp[..., 2])
    inter_y2 = torch.min(boxes1_exp[..., 3], boxes2_exp[..., 3])

    # Calculate Intersection Area
    inter_width = (inter_x2 - inter_x1).clamp(min=0)
    inter_height = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_width * inter_height  # [num_valid1, num_valid2]

    # Calculate Areas of Boxes
    area1 = (boxes1_exp[..., 2] - boxes1_exp[..., 0]) * (boxes1_exp[..., 3] - boxes1_exp[..., 1])  # [num_valid1, 1]
    area2 = (boxes2_exp[..., 2] - boxes2_exp[..., 0]) * (boxes2_exp[..., 3] - boxes2_exp[..., 1])  # [1, num_valid2]
    union_area = area1 + area2 - inter_area  # [num_valid1, num_valid2]

    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)  # [num_valid1, num_valid2]

    # Calculate Centers of Boxes
    center_x1 = (boxes1_exp[..., 0] + boxes1_exp[..., 2]) / 2
    center_y1 = (boxes1_exp[..., 1] + boxes1_exp[..., 3]) / 2
    center_x2 = (boxes2_exp[..., 0] + boxes2_exp[..., 2]) / 2
    center_y2 = (boxes2_exp[..., 1] + boxes2_exp[..., 3]) / 2

    # Calculate Center Distance
    center_dist = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2  # [num_valid1, num_valid2]

    # Calculate Smallest Enclosing Box
    enclose_x1 = torch.min(boxes1_exp[..., 0], boxes2_exp[..., 0])
    enclose_y1 = torch.min(boxes1_exp[..., 1], boxes2_exp[..., 1])
    enclose_x2 = torch.max(boxes1_exp[..., 2], boxes2_exp[..., 2])
    enclose_y2 = torch.max(boxes1_exp[..., 3], boxes2_exp[..., 3])
    enclose_dist = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2  # [num_valid1, num_valid2]

    # Calculate DIoU
    diou = iou - (center_dist / (enclose_dist + 1e-6))  # [num_valid1, num_valid2]

    return diou  # [num_valid1, num_valid2]

def postprocess_yolo_output2(output, loaded_anchor_boxes, stride=(Constants.desired_size / 16)):
    # Unpack the output shape
    batch_size, num_anchors, grid_height, grid_width, num_outputs = output.shape

    # Generate grid indices (center positions) for each grid cell
    cx = torch.arange(grid_width, device=output.device).view(1, 1, 1, grid_width, 1).expand(batch_size, num_anchors, grid_height, grid_width, 1)
    cy = torch.arange(grid_height, device=output.device).view(1, 1, grid_height, 1, 1).expand(batch_size, num_anchors, grid_height, grid_width, 1)

    # Apply sigmoid to normalize x, y coordinates
    output[..., 0:2] = torch.sigmoid(output[..., 0:2])

    # Adjust x and y coordinates relative to each grid cell
    output[..., 0] += cx.squeeze(-1)  # x-coordinates
    output[..., 1] += cy.squeeze(-1)  # y-coordinates

    # Convert x, y from grid index to absolute pixel values
    output[..., 0] *= stride
    output[..., 1] *= stride

    # Ensure anchor boxes are correctly applied
    if loaded_anchor_boxes is not None and len(loaded_anchor_boxes) == num_anchors:
        # Reshape anchors: [1, num_anchors, 1, 1, 2] -> Anchor widths and heights
        anchors = loaded_anchor_boxes.view(1, num_anchors, 1, 1, 2)

        # Split anchors into widths and heights
        anchor_widths = anchors[..., 0].expand(batch_size, num_anchors, grid_height, grid_width, 1)
        anchor_heights = anchors[..., 1].expand(batch_size, num_anchors, grid_height, grid_width, 1)

        # Apply the exponential function to predict width and height
        output[..., 2] = torch.exp(output[..., 2]) * anchor_widths * Constants.desired_size
        output[..., 3] = torch.exp(output[..., 3]) * anchor_heights * Constants.desired_size
    else:
        raise ValueError("Anchor boxes are not configured correctly.")

    # Apply sigmoid to the object confidence scores
    output[..., 4] = torch.sigmoid(output[..., 4])

    return output

def apply_greedy_matching(dious, threshold=0.5):
    """
    Greedy matching based on DIoU values.
    This function replaces the Hungarian matching with a simple greedy approach.
    
    Args:
        dious (Tensor): DIoU values of shape [N, num_valid].
        threshold (float): Threshold for matching.

    Returns:
        matched_indices (tuple): Indices of matched predictions and ground truth boxes.
    """
    # Find the best matching box for each ground truth box (greedy max)
    max_dious, pred_indices = torch.max(dious, dim=0)

    # Apply the threshold to filter out low DIoU matches
    matched_gt_indices = torch.arange(dious.size(1), device=dious.device)[max_dious > threshold]

    # Filter corresponding prediction indices
    matched_pred_indices = pred_indices[max_dious > threshold]

    return matched_pred_indices, matched_gt_indices




def postprocess_yolo_output(output, loaded_anchor_boxes):
    batch_size, num_anchors, grid_height, grid_width, num_outputs = output.shape
    stride=(Constants.desired_size/grid_height)

    cx, cy = generate_grid_indices(grid_height, grid_width, output.device)

    # Apply sigmoid to normalize x, y coordinates
    output[..., 0:2] = torch.sigmoid(output[..., 0:2])

        # Expand cx and cy to match output tensor shape
    cx = cx.unsqueeze(0).unsqueeze(0).expand(batch_size, num_anchors, grid_height, grid_width).to(output.device)
    cy = cy.unsqueeze(0).unsqueeze(0).expand(batch_size, num_anchors, grid_height, grid_width).to(output.device)

    # Adjust x and y to be the center of each cell:
    output[..., 0] += cx  # Add grid x indices for each box
    output[..., 1] += cy  # Add grid y indices for each box

    # Convert x, y from grid index to absolute pixel values
    output[..., 0] *= stride
    output[..., 1] *= stride

    # Adjust widths and heights from the anchor boxes
    if loaded_anchor_boxes is not None and len(loaded_anchor_boxes) == num_anchors:
        loaded_anchor_boxes = loaded_anchor_boxes.to(output.device)
        # Expand anchor widths and heights to match output shape [64, 5, 40, 40, 1]
        anchor_widths = loaded_anchor_boxes[:, 0].view(1, 5, 1, 1, 1).expand(output.shape[0], output.shape[1], output.shape[2], output.shape[3], 1)
        anchor_heights = loaded_anchor_boxes[:, 1].view(1, 5, 1, 1, 1).expand(output.shape[0], output.shape[1], output.shape[2], output.shape[3], 1)

        output[..., 2] = (torch.exp(output[..., 2]).unsqueeze(-1) * anchor_widths * Constants.desired_size).squeeze(-1)
        output[..., 3] = (torch.exp(output[..., 3]).unsqueeze(-1) * anchor_heights * Constants.desired_size).squeeze(-1)

    else:
        raise ValueError("Anchor boxes not configured correctly.")

    # Apply sigmoid to confidence scores
    output[..., 4] = torch.sigmoid(output[..., 4])

    return output

def generate_grid_indices(height, width, device):
    """
    Generate grid indices for each cell in the grid.
    """
    grid_y, grid_x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    return grid_x.to(device).float(), grid_y.to(device).float()

class ClampBoxCoords(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, min_val, max_val):
        ctx.save_for_backward(inputs)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return inputs.clamp(min=min_val, max=max_val)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[inputs < ctx.min_val] = 0
        grad_input[inputs > ctx.max_val] = 0
        return grad_input, None, None

def yolo_to_corners(boxes):
    """
    Converts YOLO output (x, y, width, height) to (x1, y1, x2, y2).
    
    Args:
        boxes (Tensor): Tensor of shape [N, 4], where each row contains 
                        (x, y, width, height).
                        x, y are the center coordinates, 
                        width and height are the dimensions of the box.
    
    Returns:
        Tensor: Converted boxes in (x1, y1, x2, y2) format.
    """
    # Extract center coordinates and dimensions
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    # Calculate top-left corner (x1, y1)
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    
    # Calculate bottom-right corner (x2, y2)
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    # Stack into the (x1, y1, x2, y2) format
    return torch.stack((x1, y1, x2, y2), dim=1)


def yolo_to_corners_batches(boxes):
    """
    Converts YOLO output (x_center, y_center, width, height) to (x1, y1, x2, y2)
    for batched inputs.

    Args:
        boxes (Tensor): Tensor of shape [batch_size, n, 4], where each box is represented as
                        (x_center, y_center, width, height).

    Returns:
        Tensor: Converted boxes in (x1, y1, x2, y2) format with shape [batch_size, n, 4].
    """
    # Ensure the input tensor has the correct shape
    if boxes.ndim != 3 or boxes.size(-1) != 4:
        raise ValueError("Input tensor must have shape [batch_size, n, 4]")

    # Extract center coordinates and dimensions
    x_center = boxes[..., 0]
    y_center = boxes[..., 1]
    width = boxes[..., 2]
    height = boxes[..., 3]

    # Calculate top-left corner (x1, y1)
    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)

    # Calculate bottom-right corner (x2, y2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)

    # Stack the coordinates into (x1, y1, x2, y2) format
    corners = torch.stack((x1, y1, x2, y2), dim=-1)

    return corners

def clipBoxes(pred_boxes, target_boxes):
    # Clipping coordinates using the custom ClampBoxCoords function
    clipped_pred_boxes = ClampBoxCoords.apply(pred_boxes, 0, Constants.desired_size)
    clipped_target_boxes = ClampBoxCoords.apply(target_boxes, 0, Constants.desired_size)
    return clipped_pred_boxes, clipped_target_boxes


def fix_box_coordinates(boxes, epsilon=1):
    # Ensure (x1, y1) is top-left and (x2, y2) is bottom-right
    x1 = torch.min(boxes[..., 0], boxes[..., 2])
    y1 = torch.min(boxes[..., 1], boxes[..., 3])
    x2 = torch.max(boxes[..., 0], boxes[..., 2])
    y2 = torch.max(boxes[..., 1], boxes[..., 3])

    # Only add epsilon when x1 == x2 or y1 == y2
    x2 = torch.where(x1 == x2, x1 + epsilon, x2)
    y2 = torch.where(y1 == y2, y1 + epsilon, y2)

    return torch.stack([x1, y1, x2, y2], dim=-1)

def extract_top_bboxes(pred_tensor, max_boxes=Constants.max_boxes):
    """
    Extracts the top-k bounding boxes based on confidence, padded to max_boxes size if necessary.

    Args:
        pred_tensor (Tensor): Output tensor of shape [batch_size, grid_size, grid_size, num_boxes, 5].
        max_boxes (int): Number of bounding boxes to keep.

    Returns:
        Tensor: Bounding boxes of shape [batch_size, max_boxes, 4].
        Tensor: Confidence scores of shape [batch_size, max_boxes].
    """
    batch_size, grid_size, _, num_boxes, _ = pred_tensor.shape
    cell_size = Constants.desired_size / grid_size  # Grid cell size in pixels

    # Reshape the tensor to [batch_size, grid_size * grid_size * num_boxes, 5]
    pred_tensor = pred_tensor.view(batch_size, -1, 5)

    # Separate coordinates and confidence scores
    coords = pred_tensor[..., :4]  # [batch_size, N, 4]
    confidences = pred_tensor[..., 4]  # [batch_size, N]

    # Convert grid-relative coordinates to image coordinates
    grid_offsets = torch.arange(grid_size, device=pred_tensor.device) * cell_size
    x_offsets, y_offsets = torch.meshgrid(grid_offsets, grid_offsets, indexing='ij')

    x_offsets = x_offsets.flatten().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
    y_offsets = y_offsets.flatten().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]

    x_offsets = x_offsets.repeat(1, num_boxes, 1).reshape(1, grid_size * grid_size * num_boxes, 1)
    y_offsets = y_offsets.repeat(1, num_boxes, 1).reshape(1, grid_size * grid_size * num_boxes, 1)

    # Adjust coordinates with offsets
    coords[..., 0::2] += x_offsets  # Adjust x1, x2
    coords[..., 1::2] += y_offsets

    # Sort by confidence in descending order and select top-k boxes
    topk_conf, topk_indices = confidences.topk(max_boxes, dim=1, largest=True, sorted=True)

    # Gather the corresponding top-k coordinates
    topk_coords = torch.gather(coords, 1, topk_indices.unsqueeze(-1).expand(-1, -1, 4))

    # If fewer than max_boxes, pad with zeros
    if topk_coords.size(1) < max_boxes:
        padding = max_boxes - topk_coords.size(1)
        topk_coords = torch.cat([topk_coords, torch.zeros(batch_size, padding, 4, device=pred_tensor.device)], dim=1)
        topk_conf = torch.cat([topk_conf, torch.zeros(batch_size, padding, device=pred_tensor.device)], dim=1)

    return topk_coords, topk_conf


def coordinate_penalty_loss(pred_boxes, max=Constants.desired_size):
    # Extract coordinates in (x, y, width, height) format
    x_pred, y_pred = pred_boxes[..., 0], pred_boxes[..., 1]
    width_pred, height_pred = pred_boxes[..., 2], pred_boxes[..., 3]

    # 1. Penalize invalid x and y coordinates
    negative_x_penalty = torch.relu(-x_pred)  # Penalize if x < 0
    negative_y_penalty = torch.relu(-y_pred)  # Penalize if y < 0

    excess_x_penalty = torch.relu(x_pred - max)  # Penalize if x > desired size
    excess_y_penalty = torch.relu(y_pred - max)  # Penalize if y > desired size

    # 2. Penalize invalid width and height
    zero_width_penalty = torch.relu(1 - width_pred)  # Penalize if width <= 0 (with epsilon to avoid exact 0)
    zero_height_penalty = torch.relu(1 - height_pred)  # Penalize if height <= 0 (with epsilon)

    excess_width_penalty = torch.relu(width_pred - max)  # Penalize if width > desired size
    excess_height_penalty = torch.relu(height_pred - max)  # Penalize if height > desired size

    # 3. Combine penalties into a scalar
    coordinate_penalty = (
        negative_x_penalty.mean() + negative_y_penalty.mean() +
        excess_x_penalty.mean() + excess_y_penalty.mean()
    )

    size_penalty = (
        zero_width_penalty.mean() + zero_height_penalty.mean() +
        excess_width_penalty.mean() + excess_height_penalty.mean()
    )

    # 4. Compute the final total penalty (normalized with mean)
    total_penalty = (coordinate_penalty + size_penalty).mean()/2

    return total_penalty

def confidence_penalty_loss(confidence, max=1):

    # 1. Penalize invalid x and y coordinates
    negative_confidence_penalty = torch.relu(-confidence)  # Penalize if x < 0

    if max != -1:
        excess_confidence_penalty = torch.relu(confidence - max)  # Penalize if x > desired size
    else:
        excess_confidence_penalty = 0


    # 4. Compute the final total penalty (normalized with mean)
    total_penalty = (negative_confidence_penalty + excess_confidence_penalty).mean()

    return total_penalty

def calculate_area(boxes):
    # Ensure boxes have valid dimensions (x2 > x1, y2 > y1)
    width = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    height = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    return width * height

class DIoULoss(nn.Module):
    def __init__(self):
        super(DIoULoss, self).__init__()

    def forward(self, pred_boxes, target_boxes, eps=1e-7):
        """
        DIoU loss for bounding boxes in (x1, y1, x2, y2) format.
        
        Args:
            pred_boxes (Tensor): Predicted boxes with shape [batch_size, num_boxes, 4]
            target_boxes (Tensor): Ground truth boxes with shape [batch_size, num_boxes, 4]
            eps (float): Small value to avoid division by zero.
        
        Returns:
            Tensor: DIoU loss.
        """
        iou = IoULoss.calculate_valid_iou(pred_boxes, target_boxes)

        # 5. Calculate the center points of both boxes
        pred_center_x = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2
        pred_center_y = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2
        target_center_x = (target_boxes[..., 0] + target_boxes[..., 2]) / 2
        target_center_y = (target_boxes[..., 1] + target_boxes[..., 3]) / 2

        # 6. Calculate the squared distance between the centers
        center_dist = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2

        # 7. Calculate the diagonal distance of the smallest enclosing box
        enclosing_x1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
        enclosing_y1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
        enclosing_x2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
        enclosing_y2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
        enclosing_diagonal = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2

        # 8. Calculate DIoU
        diou = iou - (center_dist / (enclosing_diagonal + eps))

        # 9. DIoU Loss (1 - DIoU)
        loss = 1 - diou

        # Return the mean loss over the batch
        return loss
    


# Define IoU loss (Inverse IoU loss)
class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred_boxes, true_boxes):
        # Reshape the tensors to combine batch and bounding box dimensions for IoU calculation
        pred_boxes_flat = pred_boxes.view(-1, 4)  # Flatten to [batch_size * num_boxes, 4]
        true_boxes_flat = true_boxes.view(-1, 4)    # Flatten to [batch_size * num_boxes, 4]
        
        # Clamp predicted values to ensure non-negative areas
        #pred_boxes_flat = torch.clamp(pred_boxes_flat, min=0, max=512)
        #target_boxes_flat = torch.clamp(target_boxes_flat, min=0, max=512)

        avg_batch_iou  = IoULoss.calculate_iou(pred_boxes_flat, true_boxes_flat)  # IoU between predicted and true boxes
        
        return 1 - avg_batch_iou


    def calculate_iou(box1, box2):
        """ Calculate IoU of single predicted and ground truth box """
        # Determine the coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The area of intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # The area of both rectangles
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Compute the union area by using Inclusion-Exclusion principle
        union_area = box1_area + box2_area - intersection_area

        # Compute the IoU
        iou = intersection_area / union_area
        return iou
    
    def calculate_valid_iou(pred_boxes, target_boxes):
        # Fix coordinates to ensure correct order and min margin
        #pred_boxes = IoULoss.fix_box_coordinates(pred_boxes)

        # Calculate IoU for all valid bounding boxes
        iou_matrix = ops.box_iou(pred_boxes, target_boxes)

        # Use max IoU per ground truth box to ensure best alignment
        max_iou, _ = iou_matrix.max(dim=0)  # Max IoU for each true box

        max_iou = torch.nan_to_num(max_iou, nan=0.0)
        max_iou = torch.clamp(max_iou, min=0.0)  # Clamp negative values to 0

        return max_iou

class ConfidencePenalty(nn.Module):
    def __init__(self):
        super(ConfidencePenalty, self).__init__()
        self.LowConfidenceScale = 1
        self.HighConfidenceScale = 0.5

        self.targetConfidence = 0.8

    def forward(self, confidences):
        # Penalize confidences less than 0
        low_confidence_penalty = torch.clamp(-confidences, min=0).mean() * self.LowConfidenceScale

        # Penalize confidences greater than 1
        high_confidence_penalty = torch.clamp(confidences - 1, min=0).mean() * self.HighConfidenceScale

        # Return the combined penalty
        return low_confidence_penalty + high_confidence_penalty


class CombinedLoss(nn.Module):
    def __init__(self, anchor_boxes):
        super(CombinedLoss, self).__init__()
        self.iou_loss = IoULoss()
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        self.diou_loss = DIoULoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.confidencePenalty = ConfidencePenalty()
        self.anchor_boxes = anchor_boxes

        #Weight of All Penalties vs All Loss functions
        #1 for penalties, 0 for Loss Functions
        self.alpha = 0.35
        


        self.IoUScale = 0 # Weight of IoU loss
        self.DIoUScale = 0.95 #Weight of DIoU loss
        self.SmoothL1LossScale = 1.4 #Weight of SmoothL1Loss
        self.BCEScale = 0.0010

        self.confidenceScale = 1.0 #Weight of our Confidence levels
        self.incorrectAreaScale = 0.1

        self.coordinate_penalty_weight = 0.0
        self.confidence_penalty_weight = 0.0

        self.negativePenaltyWeight = 0.2

        self.GoodMultiplier = 1
        self.BadMultiplier = 0

        self.outOfBoundsPenaltyScale = 0.05

    def updateAlpha(self, alpha):
        self.alpha = alpha

    def getAlpha(self):
        return self.alpha
    

    def forward(self, pred_boxes, target_boxes, writer=None, step=-1):

        pred_boxes[0] = pred_boxes[0][..., :5]
        pred_boxes[1] = pred_boxes[1][..., :5]
        pred_boxes[2] = pred_boxes[2][..., :5]

        pred_boxes[0] = postprocess_yolo_output(pred_boxes[0], self.anchor_boxes)
        pred_boxes[1] = postprocess_yolo_output(pred_boxes[1], self.anchor_boxes)
        pred_boxes[2] = postprocess_yolo_output(pred_boxes[2], self.anchor_boxes)

        batch_size, num_anchors, grid_h, grid_w, num_outputs = pred_boxes[0].shape

        pred_boxes[0] = pred_boxes[0].view(batch_size, -1, 5)  # N = num_anchors * grid_h * grid_w
        pred_boxes[1] = pred_boxes[1].view(batch_size, -1, 5)  # N = num_anchors * grid_h * grid_w
        pred_boxes[2] = pred_boxes[2].view(batch_size, -1, 5)
        
        #ConfidencePenalty = self.confidencePenalty(pred_boxes[..., 4:5])
        if False:
            print()

        #pred_width = pred_boxes[..., 2]
        #pred_height = pred_boxes[..., 3]

        #negative_width_penalty = torch.clamp(-pred_width, min=0)  # Only penalize if < 0
        #negative_height_penalty = torch.clamp(-pred_height, min=0)  # Only penalize if < 0

        pred_boxes = torch.cat(pred_boxes, dim=1)

        pred_boxes[..., :4] = yolo_to_corners_batches(pred_boxes[..., :4])

        bce = []

        for i in range(batch_size):

            temp = (calculate_target_conf(pred_boxes[i][..., :4], target_boxes[i].view(-1, 4)))

            bce.append(self.bce_loss(pred_boxes[i][..., 4], temp))

        bce_loss_value = torch.cat(bce, dim=0)

        # Calculate penalties for coordinates less than 0
        lower_bound_penalty = torch.relu(-pred_boxes[..., :4])  # Penalize values < 0
        
        # Calculate penalties for coordinates greater than desired_size
        upper_bound_penalty = torch.relu(pred_boxes[..., :4] - Constants.desired_size)  # Penalize values > desired_size
        
        # Combine penalties
        tp = ((lower_bound_penalty + upper_bound_penalty).mean()) * self.outOfBoundsPenaltyScale

        #Trim the padding bboxes, and remove the least confident bboxes for the corresponding batch Item
        target_boxes, pred_boxes, confidences_flat = filter_and_trim_boxes(pred_boxes, target_boxes)

        #difference = pred_boxes.mean() - confidences_flat.mean()


        #If for some reason there's still a 0,0,0,0, add an elipse. Flip x1,x2 and y1,y2 if x1 > x2 or y1 > y2
        target_boxes = fix_box_coordinates(target_boxes)

        if torch.isnan(pred_boxes).any() or torch.isinf(pred_boxes).any():
            print(f"NaN or Inf detected in images at step {step}")

        #pred_boxes[:, 2] = torch.clamp(pred_boxes[:, 2], min=1)  # Clamp width
        #pred_boxes[:, 3] = torch.clamp(pred_boxes[:, 3], min=1)  # Clamp height


        # Extract x1, y1, x2, y2 from pred_boxes
        #x1 = pred_boxes[..., 0]
        #y1 = pred_boxes[..., 1]
        #x2 = pred_boxes[..., 2]
        #y2 = pred_boxes[..., 3]

        # Apply ReLU-based penalty for out-of-bounds values
        #x1_penalty = torch.relu(-x1)  # Penalize if x1 < 0
        #y1_penalty = torch.relu(-y1)  # Penalize if y1 < 0
        #x2_penalty = torch.relu(x2 - Constants.desired_size)  # Penalize if x2 > desired_size
        #y2_penalty = torch.relu(y2 - Constants.desired_size)  # Penalize if y2 > desired_size

        # Combine all penalties
        outOfBoundsPenalty = 0#(torch.mean(x1_penalty ** 2 + y1_penalty ** 2 + x2_penalty ** 2 + y2_penalty ** 2)/Constants.desired_size) * self.outOfBoundsPenaltyScale




        _, target_boxes = clipBoxes(pred_boxes,target_boxes)
        #ConfidencePenalty = confidence_penalty_loss(confidences_flat, 1)
        #penalty = CoordinatePenalty + ConfidencePenalty
        #confidences_flat = torch.clamp(confidences_flat, 0, 1)




    #PENALTIES - Apply penalties before removing the least confident bboxes with the padded target_boxes

        #Add a penalty if x1<=x2 or y1<=y2 or if either is out of bounds

    
        #Additional Penalties
        #incorrect_area_penalty =  0#((torch.abs(calculate_area(pred_boxes) - calculate_area(target_boxes))).mean()/Constants.desired_size) * self.incorrectAreaScale

        

        #scale
        #penalty_scaled = ConfidencePenalty* self.alpha

        #if step != -1:
            #writer.add_scalar('Scaled Penalty', penalty_scaled, step)

        #Min the confidence levels and scale.

        #Measure overlaps. 0 is perfect, 1 is none
        #Add a penalty for non-perfect overlap
        mask = torch.ones(target_boxes.shape[0], dtype=torch.bool, device=target_boxes.device)

        pad_size = target_boxes.shape[0]-pred_boxes.shape[0]

        if pad_size > 0:
            padding = torch.zeros(pad_size, 4, device=pred_boxes.device)
            confPadding =  torch.zeros(pad_size, 1, device=pred_boxes.device)
            pred_boxes = torch.cat([pred_boxes, padding], dim=0)
            confidences_flat = torch.cat([confidences_flat, confPadding], dim=0)
            mask[-pad_size:] = False  # Update mask to indicate padded areas


        confidences_scaled = confidences_flat * self.confidenceScale
        #try:
        iou_loss_value = 0 #self.iou_loss(pred_boxes, target_boxes)

        #get loss for distance from coordinates. Divide by desired_size to scale 0-1
        smooth_l1_loss_value = (self.smooth_l1_loss(pred_boxes, target_boxes)/Constants.desired_size)


        #Formula for overlaps + distance
        diou_loss_value = self.diou_loss(pred_boxes,target_boxes)
        
        #smooth_l1_loss_value = smooth_l1_loss_value
        iou_loss_value = iou_loss_value * mask.float()
        diou_loss_value = diou_loss_value * mask.float()
        confidences_scaled = confidences_scaled.squeeze(-1) * mask.float()
        #bce_loss_value = torch.cat(bce_loss_value)

        #scale
        iou_loss_value_scaled = iou_loss_value * self.IoUScale
        diou_loss_value_scaled = diou_loss_value * self.DIoUScale
        smooth_l1_loss_scaled = (smooth_l1_loss_value * self.SmoothL1LossScale).mean(dim=1, keepdim=True).squeeze(-1) * mask
        bce_loss_scaled = bce_loss_value * self.BCEScale

        maxLoss = self.IoUScale + self.DIoUScale + self.SmoothL1LossScale + self.BCEScale

        iou_loss_value_average = iou_loss_value_scaled.sum()/mask.float().sum()
        diou_loss_value_average = diou_loss_value_scaled.sum()/mask.float().sum()
        smooth_l1_loss_average = smooth_l1_loss_scaled.sum() /mask.float().sum()
        bce_loss_average = bce_loss_scaled.mean() #/mask.float().sum()


        x = ((iou_loss_value_scaled + diou_loss_value_scaled + smooth_l1_loss_scaled).sum()/mask.float().sum()) + bce_loss_scaled.mean()
        y= confidences_scaled.sum()/mask.float().sum()


        total_loss = (
            iou_loss_value_average +
            diou_loss_value_average +
            smooth_l1_loss_average +
            bce_loss_average +
            tp
        ) #/ (self.IoUScale + self.DIoUScale + self.SmoothL1LossScale + self.BCEScale)

        
        # Loss formula
        #Distance from perfect and confident (Lower the more perfect we are)
        good_loss = torch.sqrt((0 - x)**2 + (1 - y)**2) * self.GoodMultiplier

        #Distance from Confidently wrong. (Higher the more confidently wrong.)
        bad_loss = torch.sqrt((maxLoss - x)**2 + (1 - y)**2) * self.BadMultiplier

        # Final loss calculation. Distance from perfect minus distance from opposite of perfect
        losses = good_loss - bad_loss
        
        losses = losses
        losses = torch.clamp(losses + (maxLoss * self.BadMultiplier), min=0)
        #losses = loss (self.IoUScale + self.DIoUScale + self.SmoothL1LossScale) - ((diou_loss_value_scaled + iou_loss_value_scaled + smooth_l1_loss_scaled) * confidences_scaled).mean()

        print("Losses: ", end="")
        print(iou_loss_value_average.item(), diou_loss_value_average.item(), smooth_l1_loss_average.item(), bce_loss_average.item(), end=" ")
        #Scale total losses with confidence

        negative_penalty_loss = 0 #(negative_width_penalty + negative_height_penalty).mean() * self.negativePenaltyWeight

        if writer and step != -1:
            writer.add_scalar('avg Confidence', confidences_flat.mean(), step)
            writer.add_scalar('avg Coordinate', pred_boxes.mean(), step)
            writer.add_scalar('BCE Loss', bce_loss_average.item(), step)
            #writer.add_scalar('Negative Penalty', negative_penalty_loss, step)
            #writer.add_scalar('Too Big Penalty', outOfBoundsPenalty, step)
            #writer.add_scalar('iou_loss_value', iou_loss_value_scaled.mean(), step)
            writer.add_scalar('diou_loss_value', diou_loss_value_scaled.mean(), step)
            writer.add_scalar('smooth_l1_loss_value', smooth_l1_loss_scaled.mean(), step)
            writer.add_scalar('Loss/train', total_loss.item(), step)
            writer.add_scalar('OutOfBounds', tp.item(), step)



        # Add the negative width and height penalties to the total loss

            
        # Compute the final mean loss, ignoring 0-confidence boxes
        final_loss = losses + negative_penalty_loss + outOfBoundsPenalty# + penalty_scalled  # Avoid division by 0

        # Log percentages (for debugging/monitoring)
        #total_loss = diou_loss_value_scaled + iou_loss_value_scaled + penalty_scaled
        #iouPercent = f"{iou_loss_value_scaled/(diou_loss_value_scaled + iou_loss_value_scaled + penalty_scaled) * 100:.2f}%"
        #diouPercent = f"{diou_loss_value_scaled/(diou_loss_value_scaled + iou_loss_value_scaled + penalty_scaled) * 100:.2f}%"
        #penaltyPercent = f"{penalty/(diou_loss_value_scaled + iou_loss_value_scaled + penalty_scaled) * 100:.2f}%"
        # Combined loss: weighted sum of both IoU and SmoothL1

        return total_loss
    

    def yolo_loss(self, pred_boxes, pred_conf, pred_classes, target_boxes, target_classes, target_mask):
        """
        Compute the YOLO loss for a batch of images.
        
        pred_boxes: Tensor of shape [batch_size, num_anchors, grid_h, grid_w, 4]
        pred_conf: Tensor of shape [batch_size, num_anchors, grid_h, grid_w]
        pred_classes: Tensor of shape [batch_size, num_anchors, grid_h, grid_w, num_classes]
        
        target_boxes: Tensor of shape [batch_size, max_gt_boxes, 4]
        target_classes: Tensor of shape [batch_size, max_gt_boxes]
        target_mask: Tensor indicating which predictions correspond to ground truth.
        """

        # 1. Compute Objectness Loss (BCE Loss)
        objectness_loss = F.binary_cross_entropy_with_logits(
            pred_conf, target_mask, reduction='mean'
        )

        # 2. Compute Localization Loss (DIoU or SmoothL1 Loss) for assigned boxes
        assigned_pred_boxes = pred_boxes[target_mask]
        assigned_target_boxes = target_boxes[target_mask]
        loc_loss = F.smooth_l1_loss(assigned_pred_boxes, assigned_target_boxes, reduction='mean')

        # 3. Compute Classification Loss (only for positive boxes)
        assigned_pred_classes = pred_classes[target_mask]
        assigned_target_classes = target_classes[target_mask]
        class_loss = F.cross_entropy(assigned_pred_classes, assigned_target_classes, reduction='mean')

        # 4. Combine all losses with appropriate weights
        total_loss = loc_loss + objectness_loss + class_loss

        return total_loss


