import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from customDataSet import CustomImageDataset

class myCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 256 * 256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        print(f"Shape before view: {x.shape}")
        x = x.view(64, 16 * 256 * 256)
        x = self.fc1(x)
        return x
    


def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct=0
    total=0
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

    epoch_loss = running_loss/total
    epoch_acc = 100 * correct/total
    return epoch_loss, epoch_acc
    
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct=0
    total=0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct/total
        return epoch_loss, epoch_acc
    


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        padding = [(max_wh - w) // 2, (max_wh - h) // 2, (max_wh - w) - (max_wh - w) // 2, (max_wh - h) - (max_wh - h) // 2]
        return F.pad(image, padding, 0, 'constant')




if __name__ == "__main__":
    desired_size=1024

    transform = transforms.Compose([
        SquarePad(),  # Apply padding to make the image square
        transforms.Resize((desired_size, desired_size)),  # Resize to desired square size
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Adjust mean and std for normalization
    ])

    #train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    #test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


    train_dataset=CustomImageDataset(img_dir='./backend/training_data/', transform=transform, train=True)
    test_dataset=CustomImageDataset(img_dir='./backend/training_data/', transform=transform, train=False)
    
    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    cnn_model = myCNN()
    criterion = nn.CrossEntropyLoss()
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
