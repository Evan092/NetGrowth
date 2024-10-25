import math
import torch
import torch.nn as nn
import torchvision
import torchvision.ops as ops
import Constants
from GIoULoss import GIoULoss
import torch



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


def filter_and_trim_boxes(pred_boxes, target_boxes, max_boxes=Constants.max_boxes):
    batch_size, grid_size, _, num_boxes, _ = pred_boxes.shape

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

    for i in range(batch_size):
        # Get valid target boxes for the current image
        valid_boxes = target_boxes[i][valid_mask[i]]  # Shape: [num_valid, 4]

        # Number of valid target boxes
        num_valid = valid_boxes.shape[0]

        # Sort predicted boxes by confidence (descending)
        _, topk_indices = torch.topk(pred_confidences[i], num_valid, largest=True)

        # Select the top-k most confident predicted boxes and their confidences
        filtered_pred_coords = pred_coords[i][topk_indices]  # Shape: [num_valid, 4]
        filtered_confidences_flat = pred_confidences[i][topk_indices].unsqueeze(-1)  # [num_valid, 1]

        refiltered_pred_boxes, refiltered_confidences = filter_confidences(filtered_pred_coords, filtered_confidences_flat)
        nmsfiltered_pred_boxes, nmsfiltered_confidences = apply_nms(refiltered_pred_boxes, refiltered_confidences)

        # Append the filtered results for this image
        filtered_target_boxes.append(valid_boxes)
        filtered_pred_boxes.append(nmsfiltered_pred_boxes)
        filtered_confidences.append(nmsfiltered_confidences)





    # Step 3: Concatenate results from all images in the batch
    target_boxes_flat = torch.cat(filtered_target_boxes, dim=0)  # Shape: [total_valid, 4]
    pred_boxes_flat = torch.cat(filtered_pred_boxes, dim=0)      # Shape: [total_valid, 4]
    confidences_flat = torch.cat(filtered_confidences, dim=0)    # Shape: [total_valid, 1]


    return target_boxes_flat, pred_boxes_flat, confidences_flat

def postprocess_yolo_output(output, loaded_anchor_boxes, stride=(Constants.desired_size/16)):
    batch_size, height, width, num_boxes, _ = output.shape
    cx, cy = generate_grid_indices(height, width, output.device)

    # Apply sigmoid to normalize x, y coordinates
    output[..., 0:2] = torch.sigmoid(output[..., 0:2])

    # Adjust x and y to be the center of each cell:
    output[..., 0] += cx.unsqueeze(-1)  # Add grid x indices for each box
    output[..., 1] += cy.unsqueeze(-1)  # Add grid y indices for each box

    # Convert x, y from grid index to absolute pixel values
    output[..., 0] *= stride
    output[..., 1] *= stride

    # Adjust widths and heights from the anchor boxes
    if loaded_anchor_boxes is not None and len(loaded_anchor_boxes) == num_boxes:
        # Assuming anchor boxes is a tensor of shape [5, 2] (5 anchor boxes, each with width and height)
        anchor_widths = loaded_anchor_boxes[:, 0].unsqueeze(0).unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        anchor_heights = loaded_anchor_boxes[:, 1].unsqueeze(0).unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        # Shape becomes [1, 1, 1, 5, 1] to match [batch_size, height, width, num_boxes, 1]

        # Now, expand anchor_widths to match the output dimensions
        anchor_widths = anchor_widths.expand(batch_size, height, width, num_boxes, 1)
        anchor_heights = anchor_heights.expand(batch_size, height, width, num_boxes, 1)

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
    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
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

def yolo_to_corners(boxes, anchor_boxes):
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
        self.confidencePenalty = ConfidencePenalty()
        self.anchor_boxes = anchor_boxes

        #Weight of All Penalties vs All Loss functions
        #1 for penalties, 0 for Loss Functions
        self.alpha = 0.35
        


        self.IoUScale = 1 # Weight of IoU loss
        self.DIoUScale = 0 #Weight of DIoU loss
        self.SmoothL1LossScale = 1 #Weight of SmoothL1Loss


        self.confidenceScale = 1.0 #Weight of our Confidence levels
        self.incorrectAreaScale = 0.1

        self.coordinate_penalty_weight = 0.0
        self.confidence_penalty_weight = 0.0

        self.negativePenaltyWeight = 0.2

        self.GoodMultiplier = 2
        self.BadMultiplier = 1.2

    def updateAlpha(self, alpha):
        self.alpha = alpha

    def getAlpha(self):
        return self.alpha
    

    def forward(self, pred_boxes, target_boxes, writer, step=-1):

        #ConfidencePenalty = self.confidencePenalty(pred_boxes[..., 4:5])
        if False:
            print()

        pred_width = pred_boxes[..., 2]
        pred_height = pred_boxes[..., 3]

        negative_width_penalty = torch.clamp(-pred_width, min=0)  # Only penalize if < 0
        negative_height_penalty = torch.clamp(-pred_height, min=0)  # Only penalize if < 0




        pred_boxes = postprocess_yolo_output(pred_boxes, self.anchor_boxes)


        #Trim the padding bboxes, and remove the least confident bboxes for the corresponding batch Item
        target_boxes, pred_boxes, confidences_flat = filter_and_trim_boxes(pred_boxes, target_boxes)

        #difference = pred_boxes.mean() - confidences_flat.mean()

        #If for some reason there's still a 0,0,0,0, add an elipse. Flip x1,x2 and y1,y2 if x1 > x2 or y1 > y2
        target_boxes = fix_box_coordinates(target_boxes)



        #pred_boxes[:, 2] = torch.clamp(pred_boxes[:, 2], min=1)  # Clamp width
        #pred_boxes[:, 3] = torch.clamp(pred_boxes[:, 3], min=1)  # Clamp height

        pred_boxes = yolo_to_corners(pred_boxes, self.anchor_boxes)


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
        confidences_scaled = confidences_flat * self.confidenceScale

        #Measure overlaps. 0 is perfect, 1 is none
        #Add a penalty for non-perfect overlap
        mask = torch.ones(target_boxes.shape[0], dtype=torch.bool, device=target_boxes.device)

        pad_size = target_boxes.shape[0]-pred_boxes.shape[0]

        if pad_size > 0:
            padding = torch.zeros(pad_size, 4, device=pred_boxes.device)
            confPadding =  torch.zeros(pad_size, 1, device=pred_boxes.device)
            pred_boxes = torch.cat([pred_boxes, padding], dim=0)
            confidences_scaled = torch.cat([confidences_scaled, confPadding], dim=0)
            mask[-pad_size:] = False  # Update mask to indicate padded areas

        try:
            iou_loss_value = 0 #self.iou_loss(pred_boxes, target_boxes)

            #get loss for distance from coordinates. Divide by desired_size to scale 0-1
            smooth_l1_loss_value = (self.smooth_l1_loss(pred_boxes, target_boxes)/Constants.desired_size)

            #Formula for overlaps + distance
            diou_loss_value = self.diou_loss(pred_boxes,target_boxes)
        except Exception as e:
            # Code that runs for any other exception
            print(f"An error occurred: {e}")

        
        smooth_l1_loss_value = smooth_l1_loss_value * mask.unsqueeze(1)
        iou_loss_value = iou_loss_value * mask.float()
        diou_loss_value = diou_loss_value * mask.float()

        #scale
        iou_loss_value_scaled = iou_loss_value * self.IoUScale
        diou_loss_value_scaled = diou_loss_value * self.DIoUScale
        smooth_l1_loss_scaled = (smooth_l1_loss_value * self.SmoothL1LossScale).mean(dim=1, keepdim=True)


        maxLoss = self.IoUScale + self.DIoUScale + self.SmoothL1LossScale
        x = (iou_loss_value_scaled + diou_loss_value + smooth_l1_loss_scaled)/mask.float().sum()
        y= confidences_scaled/mask.float().sum()



        # Loss formula
        good_loss = torch.sqrt((0 - x)**2 + (1 - y)**2) * self.GoodMultiplier
        bad_loss = torch.sqrt((maxLoss - x)**2 + (1 - y)**2) * self.BadMultiplier

        # Final loss calculation
        losses = good_loss - bad_loss
        
        losses = losses
        losses = torch.clamp(losses + (maxLoss * self.BadMultiplier), min=0).mean()
        #losses = loss (self.IoUScale + self.DIoUScale + self.SmoothL1LossScale) - ((diou_loss_value_scaled + iou_loss_value_scaled + smooth_l1_loss_scaled) * confidences_scaled).mean()

        
        #Scale total losses with confidence

        negative_penalty_loss = (negative_width_penalty + negative_height_penalty).mean() * self.negativePenaltyWeight

        if step != -1:
            writer.add_scalar('avg Confidence', confidences_flat.mean(), step)
            writer.add_scalar('avg Coordinate', pred_boxes.mean(), step)
            writer.add_scalar('Negative Penalty', negative_penalty_loss, step)
            #writer.add_scalar('iou_loss_value', iou_loss_value_scaled.mean(), step)
            writer.add_scalar('diou_loss_value', diou_loss_value_scaled.mean(), step)
            writer.add_scalar('smooth_l1_loss_value', smooth_l1_loss_scaled.mean(), step)
            writer.add_scalar('Loss/train', losses.item(), step)


        # Add the negative width and height penalties to the total loss

            
        # Compute the final mean loss, ignoring 0-confidence boxes
        final_loss = losses + negative_penalty_loss# + penalty_scalled  # Avoid division by 0

        # Log percentages (for debugging/monitoring)
        #total_loss = diou_loss_value_scaled + iou_loss_value_scaled + penalty_scaled
        #iouPercent = f"{iou_loss_value_scaled/(diou_loss_value_scaled + iou_loss_value_scaled + penalty_scaled) * 100:.2f}%"
        #diouPercent = f"{diou_loss_value_scaled/(diou_loss_value_scaled + iou_loss_value_scaled + penalty_scaled) * 100:.2f}%"
        #penaltyPercent = f"{penalty/(diou_loss_value_scaled + iou_loss_value_scaled + penalty_scaled) * 100:.2f}%"
        # Combined loss: weighted sum of both IoU and SmoothL1

        return final_loss
    

