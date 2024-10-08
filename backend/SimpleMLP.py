import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3*32*32, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        print(f"Shape before view: {x.shape}")
        x = x.view(-1, 32*8*8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
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

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    num_epochs = 10
    cnn_model = SimpleCNN()
    mlp_model = SimpleMLP()

    criterion = nn.CrossEntropyLoss()
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        train_loss, train_acc = train(mlp_model, train_loader, criterion, mlp_optimizer)
        test_loss, test_acc = evaluate(mlp_model, test_loader, criterion)

    for epoch in range(num_epochs):
        train_loss, train_acc = train(cnn_model, train_loader, criterion, cnn_optimizer)
        test_loss, test_acc = evaluate(cnn_model, test_loader, criterion)

    mlp_test_loss, mlp_test_acc = evaluate(mlp_model, test_loader, criterion)
    cnn_test_loss, cnn_test_acc = evaluate(cnn_model, test_loader, criterion)

    print(f"MLP Test Loss: {mlp_test_loss:.4f}, Test Accuracy: {mlp_test_acc:.2f}%")
    print(f"CNN Test Loss: {cnn_test_loss:.4f}, Test Accuracy: {cnn_test_acc:.2f}%")
