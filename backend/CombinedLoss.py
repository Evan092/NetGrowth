import math
import torch
import torch.nn as nn
import torchvision
import torchvision.ops as ops
import Constants
from GIoULoss import GIoULoss
import torch


def extract_top_bboxes(pred_tensor, max_boxes=35):
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

def trimPaddedBboxes(pred_boxes, target_boxes):
            #Filter out Padded and least confident pred_boxes before reflatten

    pred_boxes, confidences = extract_top_bboxes(pred_boxes, target_boxes.shape[1])

    #outputs = torch.clamp(outputs, min=1e-6, max=desired_size)  # Adjust to your image size
    
    # Step 1: Create the valid mask based on the sum of bbox dimensions
    valid_mask = (target_boxes.sum(dim=-1) > 0)  # Shape: [batch_size, max_boxes]

    # Step 2: Flatten the confidences for top-k selection
    confidences_flat = confidences.view(-1)  # Shape: [batch_size * num_boxes]

    # Step 3: Compute the number of valid boxes
    num_valid = valid_mask.sum().item() - 1  # Total valid boxes minus 1 for top-k

    # Step 4: Get the top-k indices from flattened confidences
    _, topk_indices = confidences_flat.topk(num_valid, dim=0, largest=True, sorted=True)

    # Step 5: Convert the flat indices back to 2D indices (batch, box) for indexing outputs and bboxes
    batch_indices = topk_indices // 35  # Divide by num_boxes per batch
    box_indices = topk_indices % 35      # Modulo gives the box index within each batch

    # Step 6: Gather the top-k outputs and bboxes
    outputs_flat = pred_boxes[batch_indices, box_indices]  # Shape: [num_valid, 4]
    bboxes_flat = target_boxes[batch_indices, box_indices]    # Shape: [num_valid, 4]
    confidences_flat = confidences[batch_indices, box_indices].view(-1, 1)  # Shape: [num_valid, 1]
    return outputs_flat, bboxes_flat, confidences_flat

def coordinate_penalty_loss(pred_boxes, lambda_penalty=0.001):
    # Extract coordinates
    x1_pred, y1_pred, x2_pred, y2_pred = pred_boxes[..., 0], pred_boxes[..., 1], pred_boxes[..., 2], pred_boxes[..., 3]
    

    # Calculate penalty for flipped coordinates
    x_penalty = torch.relu(x1_pred - x2_pred) + (x1_pred >= x2_pred+1).float()
    y_penalty = torch.relu(y1_pred - y2_pred) + (y1_pred >= y2_pred+1).float()


    # Combine penalties into a scalar
    penalty = x_penalty.sum() + y_penalty.sum()


    # Penalties for coordinates going below zero
    negative_x1_penalty = torch.relu(-x1_pred)  # Penalizes x1 < 0
    negative_y1_penalty = torch.relu(-y1_pred)  # Penalizes y1 < 0
    negative_x2_penalty = torch.relu(-x2_pred)  # Penalizes x2 < 0
    negative_y2_penalty = torch.relu(-y2_pred)  # Penalizes y2 < 0

    # Penalties for coordinates exceeding desired size
    excess_x1_penalty = torch.relu(x1_pred - Constants.desired_size)  # Penalizes x1 > desired size
    excess_y1_penalty = torch.relu(y1_pred - Constants.desired_size)  # Penalizes y1 > desired size
    excess_x2_penalty = torch.relu(x2_pred - Constants.desired_size)  # Penalizes x2 > desired size
    excess_y2_penalty = torch.relu(y2_pred - Constants.desired_size)  # Penalizes y2 > desired size

    # Combine penalties
    total_penalty = (negative_x1_penalty + negative_y1_penalty +
                    negative_x2_penalty + negative_y2_penalty +
                    excess_x1_penalty + excess_y1_penalty +
                    excess_x2_penalty + excess_y2_penalty).sum()

    return penalty + total_penalty

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
        # 1. Calculate the area of predicted and target boxes
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(0) * (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(0)
        target_area = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(0) * (target_boxes[..., 3] - target_boxes[..., 1]).clamp(0)

        # 2. Calculate the intersection coordinates
        inter_x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
        inter_y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
        inter_x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
        inter_y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])

        # 3. Calculate the intersection area
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        # 4. Calculate IoU
        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + eps)

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

        avg_batch_iou  = IoULoss.calculate_valid_iou(pred_boxes_flat, true_boxes_flat)  # IoU between predicted and true boxes
        
        return 1 - avg_batch_iou
    
    def fix_box_coordinates(boxes, epsilon=1):
        # Ensure (x1, y1) is top-left and (x2, y2) is bottom-right
        x1 = torch.min(boxes[..., 0], boxes[..., 2])
        y1 = torch.min(boxes[..., 1], boxes[..., 3])
        x2 = torch.max(boxes[..., 0], boxes[..., 2])
        y2 = torch.max(boxes[..., 1], boxes[..., 3])

        # Ensure that x2 > x1 and y2 > y1 by at least epsilon
        x2 = torch.max(x2, x1 + epsilon)
        y2 = torch.max(y2, y1 + epsilon)

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def calculate_valid_iou(pred_boxes, target_boxes):
        # Fix coordinates to ensure correct order and min margin
        pred_boxes = IoULoss.fix_box_coordinates(pred_boxes)
        target_boxes = IoULoss.fix_box_coordinates(target_boxes)

        # Calculate IoU for all valid bounding boxes
        iou_matrix = ops.box_iou(pred_boxes, target_boxes)

        # Use max IoU per ground truth box to ensure best alignment
        max_iou, _ = iou_matrix.max(dim=0)  # Max IoU for each true box

        return max_iou



class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, penalty_weight=0.015):
        super(CombinedLoss, self).__init__()
        self.iou_loss = IoULoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.diou_loss = DIoULoss()
        self.alpha = alpha  # Adjust the weight of each loss
        self.penalty_weight = penalty_weight
        self.step=1
        self.steps=500
        self.end=10000

    def updateAlpha(self, alpha):
        self.alpha = alpha

    def getAlpha(self):
        return self.alpha
    

    def forward(self, pred_boxes, target_boxes):
        # Ensure both tensors are on the same device
        pred_boxes = pred_boxes.to(target_boxes.device)

        #Trim Confidences before penalties
        pred_boxes_trimmed = pred_boxes[..., :4]
        confidences_trimmed = pred_boxes[..., 4:5]
        pred_boxes_flat = pred_boxes_trimmed.view(-1, 4)
        confidences_flat = confidences_trimmed.view(-1, 1)



    #PENALTIES - Apply penalties before removing the least confident bboxes with the padded target_boxes

        #Add a penalty if x1<=x2 or y1<=y2 or if either is out of bounds
        penalty = coordinate_penalty_loss(pred_boxes_flat,self.penalty_weight)/Constants.desired_size


        # Calculate penalty for confidences greater than 1
        lacking_confidence = torch.clamp(.8-confidences_flat, max=1)  # Only keep values > 1
        # Add Penalty for Confidence Values <= 0
        low_confidence_penalty = (lacking_confidence.sum() * 1.8)/Constants.desired_size  # Adjust penalty weight as needed

        # Calculate penalty for confidences greater than 1
        excess_confidence = torch.clamp(confidences_flat - .8, min=0)  # Only keep values > 1

        # Sum the excess confidences and scale the penalty
        high_confidence_penalty = (excess_confidence.sum() * 2) / Constants.desired_size

        


        #Trim the padding bboxes, and remove the least confident bboxes for the corresponding batcb Item
        pred_boxes, target_boxes, confidences_flat = trimPaddedBboxes(pred_boxes, target_boxes)
        

        #Additional Penalties
        incorrect_area_penalty =  ((torch.abs(calculate_area(pred_boxes) - calculate_area(target_boxes))).sum()/Constants.desired_size) * 0.1


        #scale
        penalty_scaled = (penalty + low_confidence_penalty + high_confidence_penalty + incorrect_area_penalty) * self.penalty_weight


        #Min the confidence levels:
        confidences_flat = torch.clamp(confidences_flat, min=0.1)
        confidences_scaled = confidences_flat

        #Measure overlaps. 0 is perfect, 1 is none
        #Add a penalty for non-perfect overlap
        iou_loss_value = self.iou_loss(pred_boxes, target_boxes)

        smooth_l1_loss_value = (self.smooth_l1_loss(pred_boxes, target_boxes)/Constants.desired_size)

        #Formula for overlaps + distance
        diou_loss_value = self.diou_loss(pred_boxes,target_boxes)

        #scale with alpha
        iou_loss_value_scaled = iou_loss_value * (1-self.alpha)
        diou_loss_value_scaled = diou_loss_value * self.alpha




        # Compute the final mean loss, ignoring 0-confidence boxes
        final_loss = penalty_scaled +( ((diou_loss_value_scaled + iou_loss_value_scaled) * confidences_scaled)).mean()  # Avoid division by 0

        # Log percentages (for debugging/monitoring)
        #total_loss = diou_loss_value_scaled + iou_loss_value_scaled + penalty_scaled
        #iouPercent = f"{iou_loss_value_scaled/(diou_loss_value_scaled + iou_loss_value_scaled + penalty_scaled) * 100:.2f}%"
        #diouPercent = f"{diou_loss_value_scaled/(diou_loss_value_scaled + iou_loss_value_scaled + penalty_scaled) * 100:.2f}%"
        #penaltyPercent = f"{penalty/(diou_loss_value_scaled + iou_loss_value_scaled + penalty_scaled) * 100:.2f}%"
        # Combined loss: weighted sum of both IoU and SmoothL1

        return final_loss