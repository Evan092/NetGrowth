import torch

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
    assigned_mask = torch.zeros((batch_size, num_anchors, grid_h, grid_w), dtype=torch.bool)

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

# Example usage
pred_boxes = torch.rand(2, 5, 30, 30, 4)  # [batch_size=2, anchors=5, grid_h=30, grid_w=30, 4]
target_boxes = torch.tensor([
    [[10, 10, 50, 50], [0, 0, 0, 0]],  # Image 1: 1 GT box + 1 padded
    [[20, 20, 60, 60], [40, 40, 80, 80]]  # Image 2: 2 GT boxes
])  # [batch_size=2, max_gt_boxes=2, 4]

assigned_mask = assign_boxes(pred_boxes, target_boxes)
print(assigned_mask.shape)  # Should be [2, 5, 30, 30]
