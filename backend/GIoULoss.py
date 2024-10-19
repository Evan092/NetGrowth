import torch
import torch.nn as nn
import torchvision.ops as ops

class GIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, true_boxes):
            

        # Filter out all-zero ground truth boxes
        valid_mask = (true_boxes.sum(dim=1) > 0)

        # Apply the mask to both predictions and ground truth
        pred_boxes = pred_boxes[valid_mask]
        true_boxes = true_boxes[valid_mask]

        # Find the Intersection area
        inter_x1 = torch.maximum(pred_boxes[:, 0], true_boxes[:, 0])
        inter_y1 = torch.maximum(pred_boxes[:, 1], true_boxes[:, 1])
        inter_x2 = torch.minimum(pred_boxes[:, 2], true_boxes[:, 2])
        inter_y2 = torch.minimum(pred_boxes[:, 3], true_boxes[:, 3])


        intersection = torch.clamp((inter_x2 - inter_x1), min=0) * \
                    torch.clamp((inter_y2 - inter_y1), min=0)

        # Area of predicted and true boxes
        area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                    (pred_boxes[:, 3] - pred_boxes[:, 1])
        
        area_true = (true_boxes[:, 2] - true_boxes[:, 0]) * \
                    (true_boxes[:, 3] - true_boxes[:, 1])

        # Union area
        union = area_pred + area_true - intersection
        union = torch.clamp(union, min=1e-6)  # Avoid division by zero

        # IoU calculation
        iou = intersection / union


        enclosing_area = torch.clamp((inter_x2 - inter_x1) * (inter_y2 - inter_y1), min=1e-6)

        # GIoU calculation
        giou = iou - ((enclosing_area - union) / enclosing_area)


        # Calculate IoU for all valid bounding boxes
        iou_matrix = ops.box_iou(pred_boxes, true_boxes)

                # Penalize overlapping predictions
        overlap_penalty = torch.sum(iou_matrix > 0.5, dim=1).float().mean()

        # Return GIoU Loss
        return 1 - giou.mean()
