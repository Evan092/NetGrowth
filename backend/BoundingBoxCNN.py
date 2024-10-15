import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from customDataSet import CustomImageDataset
import torchvision.ops as ops
from transforms import ResizeToMaxDimension


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BoundingBoxCnn(nn.Module):
    def __init__(self, max_boxes, maxWidth, maxHeight):
        super().__init__()
        self.max_boxes = max_boxes
        self.maxWidth = maxWidth
        self.maxHeight = maxHeight
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=2) #Channel_In, Channel_Out, Kernel_Size, Padding
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2) #Channel_In, Channel_Out, Kernel_Size, Padding
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2) #Channel_In, Channel_Out, Kernel_Size, Padding
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1) #Channel_In, Channel_Out, Kernel_Size, Padding
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1) #Channel_In, Channel_Out, Kernel_Size, Padding
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1) #Channel_In, Channel_Out, Kernel_Size, Padding
        self.fc1 = nn.Linear(32 * (self.maxWidth//2//2//2) * (self.maxHeight//2//2//2), 4*self.max_boxes)  # Channel_In*WidthAfterPool*HeightAfterPool
        self.convFinal = nn.Conv2d(256, 4 * self.max_boxes, kernel_size=1)  # Predict 4 values (x, y, width, height) for each bbox
 #Channel_In*WidthAfterPool*HeightAfterPool

    def forward(self, x):#thrt
        x = self.relu(self.conv1(x))

        #x = self.pool(x)
        self.maxWidth = int(self.maxWidth // 2)
        self.maxHeight = int(self.maxHeight // 2)

        x = self.relu(self.conv2(x))

        #x = self.pool(x)
        self.maxWidth = int(self.maxWidth // 2)
        self.maxHeight = int(self.maxHeight // 2)

        x = self.relu(self.conv3(x))

        #x = self.pool(x)
        self.maxWidth = int(self.maxWidth // 2)
        self.maxHeight = int(self.maxHeight // 2)

        #x = self.relu(self.conv4(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        #x = self.pool(x)
        #self.maxWidth = int(self.maxWidth // 2)
        #self.maxHeight = int(self.maxHeight // 2)

        x = self.convFinal(x)

        #x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]) # -1, Channel_In*WidthAfterPool*HeightAfterPool
        #x = x.view(batch_size, self.max_boxes, 4, x.shape[2], x.shape[3])  # Reshape into [batch_size, max_boxes, 4, Imgwidth, Imgheight]
        x = torch.mean(x, dim=[2, 3])  # Perform global average pooling over the height and width to get [batch_size, max_boxes, 4]
        #x = self.fc1(x)
        x = x.view(-1, self.max_boxes, 4)
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
    
def train(model, loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    total = 0
    total_iou = 0.0
    for images, bboxes in loader:
        # Forward pass: compute model outputs (bounding box coordinates)
        images = images.to(device)
        bboxes = bboxes.to(device)
        outputs = model(images)
        
        # Compute the loss between predicted and true bounding box coordinates
        loss = criterion(outputs, bboxes)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the loss for the current batch
        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        # Calculate IoU for the batch
        # Reshape the tensors to combine batch and bounding box dimensions for IoU calculation
        outputs_flat = outputs.view(-1, 4)  # Flatten to [batch_size * num_boxes, 4]
        bboxes_flat = bboxes.view(-1, 4)    # Flatten to [batch_size * num_boxes, 4]

        avg_batch_iou  = calculate_valid_iou(outputs_flat, bboxes_flat)  # IoU between predicted and true boxes
        #avg_batch_iou = batch_iou.diag().mean().item()  # Average IoU for the batch

        total_iou += avg_batch_iou * images.size(0)

    # Average loss and IoU per epoch
    epoch_loss = running_loss / total
    epoch_iou = total_iou / total
    return epoch_loss, epoch_iou


def evaluate(model, loader, criterion):
    model.eval()  # Set the model to eval mode
    running_loss = 0.0
    total = 0
    total_iou=0.0
    with torch.no_grad():
        for images, bboxes in loader:
            # Forward pass: compute model outputs (bounding box coordinates)
            images = images.to(device)
            bboxes = bboxes.to(device)
            outputs = model(images)
            
            # Compute the loss between predicted and true bounding box coordinates
            loss = criterion(outputs, bboxes)

            # Backward pass and optimization
            optimizer.zero_grad()
            optimizer.step()

            # Accumulate the loss for the current batch
            running_loss += loss.item() * images.size(0)
            total += images.size(0)

            # Calculate IoU for the batch
            # Reshape the tensors to combine batch and bounding box dimensions for IoU calculation
            outputs_flat = outputs.view(-1, 4)  # Flatten to [batch_size * num_boxes, 4]
            bboxes_flat = bboxes.view(-1, 4)    # Flatten to [batch_size * num_boxes, 4]

            avg_batch_iou  = calculate_valid_iou(outputs_flat, bboxes_flat)  # IoU between predicted and true boxes
            #avg_batch_iou = batch_iou.diag().mean().item()  # Average IoU for the batch

            total_iou += avg_batch_iou * images.size(0)

        # Average loss and IoU per epoch
        epoch_loss = running_loss / total
        epoch_iou = total_iou / total
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
    valid_mask = (true_boxes[:, 2] > true_boxes[:, 0]) & (true_boxes[:, 3] > true_boxes[:, 1])

    # Filter out the padding boxes
    valid_pred_boxes = pred_boxes[valid_mask]
    valid_true_boxes = true_boxes[valid_mask]

    if valid_pred_boxes.size(0) == 0 or valid_true_boxes.size(0) == 0:
        return 0.0  # If no valid boxes, return IoU of 0

    # Calculate IoU only for valid bounding boxes
    iou_matrix = torchvision.ops.box_iou(valid_pred_boxes, valid_true_boxes)
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


    

if __name__ == "__main__":
    desired_size=1024




    transform = transforms.Compose([
        ResizeToMaxDimension(max_dim=1024),  # Resize based on max dimension while maintaining aspect ratio
        #transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3476, 0.3207, 0.2946], std=[0.3005, 0.2861, 0.2745])
    ])

#Mean: tensor([0.3476, 0.3207, 0.2946]), Std: tensor([0.3005, 0.2861, 0.2745])


    #train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    #test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


    train_dataset=CustomImageDataset(img_dir='./backend/training_data/', transform=transform, train=True)
    test_dataset=CustomImageDataset(img_dir='./backend/training_data/', transform=transform, train=False)

    train_dataset.setMaxDimensions(1024, 1024)
    test_dataset.setMaxDimensions(1024, 1024)


    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    # Compute max_boxes from both training and test datasets
    #max_boxes_train = compute_max_boxes(train_loader)
    #max_boxes_test = compute_max_boxes(test_loader)
    #max_boxes = max(max_boxes_train, max_boxes_test)
    max_boxes = 35

    train_dataset.setMaxBBoxes(max_boxes)
    test_dataset.setMaxBBoxes(max_boxes)

    #mean, std = get_mean_std_RGB(train_loader)
    #print(f"Mean: {mean}, Std: {std}")

    cnn_model = BoundingBoxCnn(max_boxes,train_dataset.maxWidth, train_dataset.maxHeight).to(device)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.SmoothL1Loss().to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.0001) #, weight_decay=5e-4

    num_epochs = 100

    for epoch in range(num_epochs):
        train_loss, train_acc = train(cnn_model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(cnn_model, test_loader, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
            f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
   
    test_loss, test_acc = evaluate(cnn_model, test_loader, criterion)
    print("\nFinal Evaluation on Test Set:")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
