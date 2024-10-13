import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
#from backend.SimpleMLP import SimpleCNN
from customDataSet import CustomImageDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)



def train(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    all_labels = []  # To store all true labels
    all_preds = []   # To store all predictions

    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.numpy())
        all_preds.extend(predicted.numpy())


    epoch_loss = running_loss / total

    epoch_acc = 100 * correct / total

    epoch_precision = precision_score(all_labels, all_preds, average='macro')
    epoch_recall = recall_score(all_labels, all_preds, average='macro')

    epoch_f1 = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1

    
def evaluate(model, loader, criterion):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []  # To store all true labels
    all_preds = []   # To store all predictions

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    epoch_precision = precision_score(all_labels, all_preds, average='macro')
    epoch_recall = recall_score(all_labels, all_preds, average='macro')
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1

    


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        padding = [(max_wh - w) // 2, (max_wh - h) // 2, (max_wh - w) - (max_wh - w) // 2, (max_wh - h) - (max_wh - h) // 2]
        return F.pad(image, padding, 0, 'constant')




if __name__ == "__main__":
    desired_size=1024

    transform = transforms.Compose([
        #SquarePad(),  # Apply padding to make the image square
        #transforms.Resize((desired_size, desired_size)),  # Resize to desired square size
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Adjust mean and std for normalization
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


    #train_dataset=CustomImageDataset(img_dir='./backend/training_data/train', transform=transform, train=True)
    #test_dataset=CustomImageDataset(img_dir='./backend/training_data/test', transform=transform, train=False)
    
    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    #cnn_model = SimpleCNN()
    resnet_model = ResNet18()

    criterion = nn.CrossEntropyLoss()
    #cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)


    num_epochs = 5
    name = "RESNET18: "
    for epoch in range(num_epochs):
        train_loss, train_acc, train_precision, train_recall, train_f1 = train(resnet_model, train_loader, criterion, resnet_optimizer)

        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(resnet_model, test_loader, criterion)

        print(f'[{name}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Precision: {train_precision:.4f}, '
            f'Recall: {train_recall:.4f}, F1: {train_f1:.4f}')

        print(f'[{name}] Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Precision: {test_precision:.4f}, Recall: '
            f'{test_recall:.4f}, F1: {test_f1:.4f}')

