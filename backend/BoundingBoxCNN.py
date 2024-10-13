import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from customDataSet import CustomImageDataset
import torchvision.ops as ops

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BoundingBoxCnn(nn.Module):
    def __init__(self, max_boxes, maxWidth, maxHeight):
        super().__init__()
        self.max_boxes = max_boxes
        self.maxWidth = maxWidth
        self.maxHeight = maxHeight
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1) #Channel_In, Channel_Out, Kernel_Size, Padding
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) #Channel_In, Channel_Out, Kernel_Size, Padding
        self.fc1 = nn.Linear(16 * int((maxWidth // 2) // 2) * int((maxHeight // 2) // 2), 4*max_boxes)  # Channel_In*WidthAfterPool*HeightAfterPool
 #Channel_In*WidthAfterPool*HeightAfterPool

    def forward(self, x):#thrt
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * int((self.maxWidth // 2) // 2) * int((self.maxHeight // 2) // 2)) # -1, Channel_In*WidthAfterPool*HeightAfterPool
        x = self.fc1(x)
        x = x.view(-1, self.max_boxes, 4)
        return x
    

max_boxes = 0

def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    iou_threshold = 0.5  # IoU threshold for determining a correct prediction

    for images, bboxes in loader:
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
            
            # Count as correct if IoU > threshold for any predicted-true pair
            correct_boxes = (iou_matrix > iou_threshold).sum().item()
            correct += correct_boxes
            
            # The total number of boxes for accuracy calculation
            total += ground_truth_boxes.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total  # Calculate accuracy as a percentage
    return epoch_loss, epoch_acc
    
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct=0
    total=0
    iou_threshold = 0.5  # IoU threshold for determining a correct prediction
    with torch.no_grad():
        for images, bboxes in loader:
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
                
                # Count as correct if IoU > threshold for any predicted-true pair
                correct_boxes = (iou_matrix > iou_threshold).sum().item()
                correct += correct_boxes
                
                # The total number of boxes for accuracy calculation
                total += ground_truth_boxes.size(0)

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total  # Calculate accuracy as a percentage
        return epoch_loss, epoch_acc
    


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

class ResizeToMaxDimension:
    def __init__(self, max_dim=1024):
        self.max_dim = max_dim

    def __call__(self, image):
        # Get the original image dimensions
        width, height = image.size

        # Determine scaling factor to maintain aspect ratio
        if max(width, height) > self.max_dim:
            scale = self.max_dim / max(width, height)
            new_size = (int(width * scale), int(height * scale))  # Maintain aspect ratio
        else:
            # If the image is already smaller than the max dimension, no resizing
            new_size = (width, height)

        # Resize the image while preserving the aspect ratio
        return F.resize(image, new_size)

if __name__ == "__main__":
    desired_size=1024




    transform = transforms.Compose([
        ResizeToMaxDimension(max_dim=1024),  # Resize based on max dimension while maintaining aspect ratio
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    #train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    #test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


    train_dataset=CustomImageDataset(img_dir='./backend/training_data/', transform=transform, train=True)
    test_dataset=CustomImageDataset(img_dir='./backend/training_data/', transform=transform, train=False)

    train_dataset.setMaxDimensions(1024, 1024)
    test_dataset.setMaxDimensions(1024, 1024)


    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)



    # Compute max_boxes from both training and test datasets
    #max_boxes_train = compute_max_boxes(train_loader)
    #max_boxes_test = compute_max_boxes(test_loader)
    #max_boxes = max(max_boxes_train, max_boxes_test)
    max_boxes = 25

    train_dataset.setMaxBBoxes(max_boxes)
    test_dataset.setMaxBBoxes(max_boxes)

    cnn_model = BoundingBoxCnn(max_boxes,train_dataset.maxWidth, train_dataset.maxHeight)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

    num_epochs = 5

    for epoch in range(num_epochs):
        train_loss, train_acc = train(cnn_model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(cnn_model, test_loader, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
            f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        test_loss, test_acc = evaluate(cnn_model, test_loader, criterion)
        print("\nFinal Evaluation on Test Set:")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
