import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from GIoULoss import GIoULoss
from customDataSet import CustomImageDataset
import torchvision.ops as ops
from transforms import ResizeToMaxDimension
from datetime import datetime
import torch.profiler
from CombinedLoss import CombinedLoss
import Constants
import math
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BoundingBoxCnn(nn.Module):
    def __init__(self, max_boxes, B=2):
        super().__init__()
        self.max_boxes = max_boxes
        self.maxWidth = Constants.desired_size
        self.maxHeight = Constants.desired_size
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2,2)
        self.B = B
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2)  # 512 -> 256
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)  # 256 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)  # 128 -> 64
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)  # 64 -> 32
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)  # 32 -> 16
        #self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2)  # 16 -> 8

        self.convFinal = nn.Conv2d(512, 5*B, kernel_size=1)  # Predict 4 values (x, y, x2, y2, confidence) for each bbox

    def forward(self, x):#thrt
        #print(str(x.shape))
        x = self.relu(self.conv1(x))
        #print(str(x.shape))
        x = self.relu(self.conv2(x))
        #print(str(x.shape))
        x = self.relu(self.conv3(x))
        #print(str(x.shape))
        x = self.relu(self.conv4(x))
        #print(str(x.shape))
        x = self.relu(self.conv5(x))
        #print(str(x.shape))
        #x = self.relu(self.conv6(x))
        #print(str(x.shape))
        x = self.convFinal(x)  # [batch_size, max_boxes * 4, H, W]
        #print(str(x.shape))
        x = x.view(-1, x.shape[2], x.shape[3], self.B, 5)
        return x
    

max_boxes = 0

def train2(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    iou_threshold = 0.5  # IoU threshold for determining a correct prediction

    for images, bboxes in loader:
        images = images.to(device)
        bboxes = bboxes.to(device)
        outputs = model(images)

        # Calculate loss between predicted and ground truth bounding boxes
        loss_bbox = criterion(outputs, bboxes)

        optimizer.zero_grad()
        loss_bbox.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss_bbox.item() * images.size(0)
        
        # Loop over each image in the batch
        for i in range(outputs.size(0)):  # Iterate over batch size
            predicted_boxes = outputs[i]  # Predicted boxes for this image
            ground_truth_boxes = bboxes[i]  # Ground truth boxes for this image
            
            # Compute IoU for this image's predicted and ground truth boxes
            iou_matrix = ops.box_iou(predicted_boxes, ground_truth_boxes)
            
            # For each ground truth box, find the predicted box with the highest IoU
            max_iou_per_gt, _ = torch.max(iou_matrix, dim=0)  # Max IoU for each ground truth box
            
            # Count ground truth boxes that have a matching predicted box with IoU > threshold
            correct_boxes = (max_iou_per_gt > iou_threshold).sum().item()
            correct += correct_boxes
            
            # The total number of ground truth boxes (for accuracy calculation)
            total += ground_truth_boxes.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total  # Calculate accuracy as a percentage
    return epoch_loss, epoch_acc


def evaluate2(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct=0
    total=0
    iou_threshold = 0.5  # IoU threshold for determining a correct prediction
    with torch.no_grad():
        for images, bboxes in loader:
            images = images.to(device)
            bboxes = bboxes.to(device)
            outputs = model(images)

            # Calculate loss between predicted and ground truth bounding boxes
            loss_bbox = criterion(outputs, bboxes)

            optimizer.zero_grad()
            optimizer.step()

            # Accumulate loss
            running_loss += loss_bbox.item() * images.size(0)
            
            # Loop over each image in the batch
            for i in range(outputs.size(0)):  # Iterate over batch size
                predicted_boxes = outputs[i]  # Predicted boxes for this image
                ground_truth_boxes = bboxes[i]  # Ground truth boxes for this image
                
                # Compute IoU for this image's predicted and ground truth boxes
                iou_matrix = ops.box_iou(predicted_boxes, ground_truth_boxes)
                
                # For each ground truth box, find the predicted box with the highest IoU
                max_iou_per_gt, _ = torch.max(iou_matrix, dim=0)  # Max IoU for each ground truth box
                
                # Count ground truth boxes that have a matching predicted box with IoU > threshold
                correct_boxes = (max_iou_per_gt > iou_threshold).sum().item()
                correct += correct_boxes
                
                # The total number of ground truth boxes (for accuracy calculation)
                total += ground_truth_boxes.size(0)

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total  # Calculate accuracy as a percentage
        return epoch_loss, epoch_acc
    



def extract_top_bboxes2(pred_tensor, max_boxes=35):
    """
    Extracts the top-k bounding boxes based on confidence, padded to max_boxes size if necessary.

    Args:
        pred_tensor (Tensor): Output tensor of shape [batch_size, 16, 16, 2, 5].
        max_boxes (int): Number of bounding boxes to keep.

    Returns:
        Tensor: Bounding boxes of shape [batch_size, max_boxes, 4].
    """
    batch_size, grid_size, _, num_boxes, _ = pred_tensor.shape
    cell_size = Constants.desired_size / grid_size  # Size of each grid cell in pixels

    all_bboxes = []  # Store top-k bounding boxes for each image

    for batch_idx in range(batch_size):
        bboxes = []

        # Iterate over each grid cell
        for i in range(grid_size):
            for j in range(grid_size):
                # For each box in the cell
                for b in range(num_boxes):
                    x1, y1, x2, y2, conf = pred_tensor[batch_idx, i, j, b]

                    # Convert grid coordinates to image coordinates
                    x1 = (j + x1) * cell_size  # Horizontal offset
                    y1 = (i + y1) * cell_size  # Vertical offset
                    x2 = (j + x2) * cell_size
                    y2 = (i + y2) * cell_size

                    # Append (x1, y1, x2, y2, confidence) to list
                    bboxes.append((x1, y1, x2, y2, conf))

        # Sort all bounding boxes by confidence in descending order
        bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)

        # Keep only the top-k bounding boxes
        top_bboxes = [bbox[:4] for bbox in bboxes[:max_boxes]]  # List of [x1, y1, x2, y2]
        confidences = [bbox[4] for bbox in bboxes[:max_boxes]]  # List of confidence scores

        # Pad with [0, 0, 0, 0] if fewer than max_boxes boxes are found
        while len(top_bboxes) < max_boxes:
            top_bboxes.append([0.0, 0.0, 0.0, 0.0])
            confidences.append(0.0)

        all_bboxes.append(top_bboxes)

    # Convert the result to a tensor of shape [batch_size, max_boxes, 4]
    return torch.tensor(all_bboxes, dtype=torch.float32, device=pred_tensor.device, requires_grad=True), torch.tensor(confidences, dtype=torch.float32, device=pred_tensor.device, requires_grad=True)



    
def train(model, loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    total = 0
    total_iou = 0.0
    i = 0
    iou_threshold=0.5
    #for images, bboxes in loader:
    for batch_idx, (images, bboxes) in enumerate(loader):
        # Forward pass: compute model outputs (bounding box coordinates)
        i+=1
        percentage = (i / len(loader)) * 100
        # This checks if the current percentage point is approximately a multiple of 5
        if int(percentage) % 10 == 0:  # Ensuring that it checks every 5% increment
            print(f"{percentage:.2f}% ", end="", flush=True)
        images = images.to(device)
        bboxes = bboxes.to(device)
        outputs = model(images)



        if False:
            print("")
        # Compute the loss between predicted and true bounding box coordinates
        loss = criterion(outputs, bboxes, writer, epoch * len(train_loader) + batch_idx)

        # Log training loss for this batch to TensorBoard
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch * len(train_loader) + batch_idx)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the loss for the current batch
        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        # Calculate IoU for the batch
        outputs_trimmed = outputs[..., :4]
        outputs_flat = outputs_trimmed.view(-1, 4)
        avg_batch_iou  = 0#calculate_valid_iou(outputs_flat, bboxes.view(-1,4))  # IoU between predicted and true boxes
        #avg_batch_iou = batch_iou.diag().mean().item()  # Average IoU for the batch
        #writer.add_scalar('avg iou', avg_batch_iou, epoch * len(train_loader) + batch_idx)

        total_iou += avg_batch_iou * images.size(0)

    print()
    # Average loss and IoU per epoch
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.norm()}")
    epoch_loss = running_loss / total
    epoch_iou = 100 * (total_iou / total)
    return epoch_loss, epoch_iou


def evaluate(model, loader, criterion):
    model.eval()  # Set the model to eval mode
    running_loss = 0.0
    total = 0
    total_iou=0.0
    i=0
    iou_threshold=0.5
    with torch.no_grad():
        for images, bboxes in loader:
            # Forward pass: compute model outputs (bounding box coordinates)
            i+=1
            percentage = (i / len(loader)) * 100
            # This checks if the current percentage point is approximately a multiple of 5
            if int(percentage) % 10 == 0:  # Ensuring that it checks every 5% increment
                print(f"{percentage:.2f}% ", end="", flush=True)
            images = images.to(device)
            bboxes = bboxes.to(device)
            outputs = model(images)

            if False:
                print("")
            # Compute the loss between predicted and true bounding box coordinates
            loss = criterion(outputs, bboxes, writer)

            # Accumulate the loss for the current batch
            running_loss += loss.item() * images.size(0)
            total += images.size(0)


            avg_batch_iou  = 0#calculate_valid_iou(outputs_flat, bboxes_flat)  # IoU between predicted and true boxes
            #avg_batch_iou = batch_iou.diag().mean().item()  # Average IoU for the batch

            total_iou += avg_batch_iou * images.size(0)

        # Average loss and IoU per epoch
        epoch_loss = running_loss / total
        epoch_iou = 100 * (total_iou / total)
        return epoch_loss, epoch_iou

def get_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)  # Batch size (number of images in batch)
        images = images.view(batch_samples, images.size(1), -1)  # Reshape to [batch_size, channels, pixels]
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean, std

def get_mean_std_RGB(loader):
    mean = torch.zeros(3)  # Initialize mean for 3 channels: R, G, B
    std = torch.zeros(3)   # Initialize std for 3 channels: R, G, B
    total_images_count = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)  # Batch size (number of images in the batch)
        images = images.view(batch_samples, images.size(1), -1)  # Reshape to [batch_size, channels, pixels]
        
        # Calculate mean and std for each channel (R, G, B)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        
        total_images_count += batch_samples

    # Normalize mean and std over the entire dataset
    mean /= total_images_count
    std /= total_images_count
    
    return mean, std

def get_mean_std_RGB2(loader):
    channels_sum = torch.zeros(3)  # For RGB, we need a mean/std for each of the 3 channels
    channels_squared_sum = torch.zeros(3)
    total_images_count = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)  # Batch size (number of images in batch)
        images = images.view(batch_samples, images.size(1), -1)  # Reshape to [batch_size, channels, pixels]
        
        channels_sum += images.mean(2).sum(0)  # Sum the means for each channel
        channels_squared_sum += (images ** 2).mean(2).sum(0)  # Sum of squares for each channel
        total_images_count += batch_samples
    
    # Calculate mean and std for each channel
    mean = channels_sum / total_images_count
    std = torch.sqrt((channels_squared_sum / total_images_count) - (mean ** 2))
    
    return mean, std


def calculate_valid_iou(pred_boxes, true_boxes):
    # Create a mask to ignore [0, 0, 0, 0] padded boxes
    #valid_mask = (true_boxes[:, 2] > true_boxes[:, 0]) & (true_boxes[:, 3] > true_boxes[:, 1])

    # Filter out all-zero ground truth boxes
    valid_mask = (true_boxes.sum(dim=1) > 0)

    # Apply the mask to both predictions and ground truth
    pred_boxes = pred_boxes[valid_mask]
    true_boxes = true_boxes[valid_mask]

    if pred_boxes.size(0) == 0 or true_boxes.size(0) == 0:
        return 0.0  # If no valid boxes, return IoU of 0

    # Calculate IoU only for valid bounding boxes
    iou_matrix = torchvision.ops.box_iou(pred_boxes, true_boxes)
    diag_iou = iou_matrix.diag()  # Get the IoU for corresponding boxes

    return diag_iou.mean().item()  # Return the average IoU




class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        padding = [(max_wh - w) // 2, (max_wh - h) // 2, (max_wh - w) - (max_wh - w) // 2, (max_wh - h) - (max_wh - h) // 2]
        return F.pad(image, padding, 0, 'constant')



def collate_fn(batch):
    images, bboxes = zip(*batch)
    images = torch.stack(images, dim=0)

    # Find the max number of bounding boxes in the batch
    max_boxes = max([len(bbox) for bbox in bboxes])

    # Pad bounding boxes to match the largest number of boxes in the batch
    padded_bboxes = [torch.cat([bbox, torch.zeros(max_boxes - len(bbox), 4)], dim=0) for bbox in bboxes]

    bboxes = torch.stack(padded_bboxes, dim=0)

    return images, bboxes


def compute_max_boxes(dataloader):
    max_boxes = 0
    for _, bboxes in dataloader:
        # Assuming bboxes is a tensor of shape (batch_size, num_boxes, box_dimensions)
        # We check the size of the second dimension (num_boxes)
        for bbox_tensor in bboxes:
            num_boxes = bbox_tensor.shape[0]  # Get the number of boxes in this image
            if num_boxes > max_boxes:
                max_boxes = num_boxes
    return max_boxes

weight_decay = 1e-5
learning_rate = 0.001
alpha=.5
batch_size = 64
desired_size=Constants.desired_size
writer = ""



if __name__ == "__main__":
            num=1
            while os.path.exists('runs/YOLO v'+str(num)+' Lr'+str(learning_rate) + " wd" + str(weight_decay) + " a" + str(alpha) + " bs"+str(batch_size)):
                num += 1


            writer = SummaryWriter('runs/YOLO v'+str(num)+' Lr'+str(learning_rate) + " wd" + str(weight_decay) + " a" + str(alpha) + " bs"+str(batch_size))
            print(writer.log_dir)
        #learning_rate = 0.001
        #for j in range(4):
            #if j%2==1:
                #learning_rate = learning_rate/2
            #else:
                #learning_rate = learning_rate/5

            #writer = SummaryWriter('runs/Lr'+str(learning_rate) + " wd" + str(weight_decay) + " a" + str(alpha) + " bs"+str(batch_size))

            transform = transforms.Compose([
                ResizeToMaxDimension(max_dim=desired_size),  # Resize based on max dimension while maintaining aspect ratio
                #transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3476, 0.3207, 0.2946], std=[0.3005, 0.2861, 0.2745])
            ])

        #Mean: tensor([0.3476, 0.3207, 0.2946]), Std: tensor([0.3005, 0.2861, 0.2745])


            #train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            #test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

            torch.set_printoptions(sci_mode=False, precision=4)
            

            train_dataset=CustomImageDataset(img_dir='./backend/training_data/', transform=transform, train=True)
            test_dataset=CustomImageDataset(img_dir='./backend/training_data/', transform=transform, train=False)

            train_dataset.setMaxDimensions(desired_size, desired_size)
            test_dataset.setMaxDimensions(desired_size, desired_size)

            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,prefetch_factor=2,persistent_workers=True, pin_memory=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2,prefetch_factor=2,persistent_workers=True, pin_memory=True)

            # Compute max_boxes from both training and test datasets
            #max_boxes_train = compute_max_boxes(train_loader)
            #max_boxes_test = compute_max_boxes(test_loader)
            #max_boxes = max(max_boxes_train, max_boxes_test)
            max_boxes = Constants.max_boxes

            train_dataset.setMaxBBoxes(max_boxes)
            test_dataset.setMaxBBoxes(max_boxes)

            #mean, std = get_mean_std_RGB(train_loader)
            #print(f"Mean: {mean}, Std: {std}")

            cnn_model = BoundingBoxCnn(max_boxes).to(device)
            max_norm = 5
            #criterion = nn.CrossEntropyLoss()
            criterion = CombinedLoss().to(device)#nn.SmoothL1Loss().to(device)#CombinedLoss().to(device)
            optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay) #, weight_decay=5e-4
            # Warm-up scheduler for the first 10 epochs
            #warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=10)

            # ReduceLROnPlateau for long-term control
            plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, threshold=0.01)

            # Combine both schedulers
            #scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, plateau_scheduler], milestones=[10])
            
            num_epochs = 100
            print()
            epoch = 0


            print(f"Batch_Size={batch_size}")
            print(f"max_boxes={max_boxes}")
            print(f"weight_decay={weight_decay}")
            print(f"learning_rate={learning_rate}")
            print(f"max_norm={max_norm}")
            print(f"desired_size={desired_size}")
            print(f"alpha={alpha}")
            #print("Loss Function: (alpha*diou_loss_value+(1-alpha)*(smooth_l1_loss_value/desired_size))*(desired_size/2)")
            print(f"Using device: {device}")
            print(f"Is CUDA available: {torch.cuda.is_available()}")

            #test_loss, test_acc = evaluate(cnn_model, test_loader, criterion)
            #print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

            while epoch < num_epochs:
                #torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=max_norm)
                print(f'==========Epoch [{epoch+1}/{num_epochs}] =========')
                print(f'Training Progress ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")}):')
                train_loss, train_acc = train(cnn_model, train_loader, criterion, optimizer)
                print(f'Evaluation Progress ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")}):')
                test_loss, test_acc = evaluate(cnn_model, test_loader, criterion)
                print(f'Finished. ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})\n'
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                    f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

                print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}', end=" & ")

                # Update learning rate based on test loss
                plateau_scheduler.step(test_loss)

                # Print the updated learning rate
                print(f'New Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

                #print(f'Current Alpha: {alpha}', end=" & ")
                #alpha = min(alpha + ((1-alpha)/20), 1)
                #criterion.updateAlpha(alpha)
                #print(f'New Alpha: {alpha}')

                print(f".....................................................{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
                # After the last epoch, ask if the user wants to add more epochs
                if epoch == num_epochs - 1:
                    user_input = input("Training completed. Would you like to add more epochs? (yes/no): ").strip().lower()
                    if user_input == 'yes':
                        num_epochs += int(input("Please enter a number: "))  # Add 10 more epochs
                    else:
                        print(f"Training finished after {epoch+1} epochs.")
                epoch += 1


            del test_loader
            del train_loader
            del train_dataset
            del test_dataset
            del cnn_model
            del criterion
            del optimizer
            del writer
            gc.collect()
            #test_loss, test_acc = evaluate(cnn_model, test_loader, criterion)
            #print("\nFinal Evaluation on Test Set:")
            #print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
