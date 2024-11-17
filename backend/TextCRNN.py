import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import json
import os
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import Constants
from transforms import ResizeToMaxDimension
from customDataSet2 import CustomImageDataset2
from PIL import Image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class LabelEncoder:
    def __init__(self, char_set):
        self.char_set = char_set
        self.char2idx = {char: idx + 1 for idx, char in enumerate(char_set)}  # Start indices from 1
        self.idx2char = {idx + 1: char for idx, char in enumerate(char_set)}
        self.blank_label = 0  # CTC requires a blank label at index 0

    def encode(self, texts):
        # Encode a list of texts to a list of tensors
        lengths = []
        encoded_texts = []
        for text in texts:
            encoded = [self.char2idx[char] for char in text if char in self.char2idx]
            lengths.append(len(encoded))
            encoded_texts.append(torch.tensor(encoded, dtype=torch.long))
        return encoded_texts, lengths

    def decode(self, preds):
        # preds: Tensor of shape [batch_size, seq_len]
        pred_texts = []
        for pred in preds:
            pred_text = ''
            prev_idx = None
            for idx in pred:
                idx = idx.item()
                if idx != self.blank_label and idx != prev_idx:
                    pred_text += self.idx2char.get(idx, '')
                prev_idx = idx
            pred_texts.append(pred_text)
        return pred_texts
    
class CRNN(nn.Module):
    def __init__(self, num_classes, img_h=32, nc=1, leaky_relu=False):
        super(CRNN, self).__init__()
        self.num_classes = num_classes
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn_layers = []
        def conv_relu(i, batch_normalization=False):
            n_in = nc if i == 0 else nm[i - 1]
            n_out = nm[i]
            cnn_layers.append(nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
            if batch_normalization:
                cnn_layers.append(nn.BatchNorm2d(n_out))
            if leaky_relu:
                cnn_layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn_layers.append(nn.ReLU(inplace=True))

        conv_relu(0)
        cnn_layers.append(nn.MaxPool2d(2, 2))  # 64x16x64
        conv_relu(1)
        cnn_layers.append(nn.MaxPool2d(2, 2))  # 128x8x32
        conv_relu(2, True)
        conv_relu(3)
        cnn_layers.append(nn.MaxPool2d((2, 1), (2, 1)))  # 256x4x32
        conv_relu(4, True)
        conv_relu(5)
        cnn_layers.append(nn.MaxPool2d((2, 1), (2, 1)))  # 512x2x32
        conv_relu(6, True)

        self.cnn = nn.Sequential(*cnn_layers)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, num_classes)
        )

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "Height after conv layers must be 1"
        conv = conv.squeeze(2)  # Remove the height dimension
        conv = conv.permute(2, 0, 1)  # [width, batch_size, channels]
        output = self.rnn(conv)
        # Output shape: [seq_len, batch_size, num_classes]
        return output

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)  # *2 because bidirectional

    def forward(self, input):
        # input: [seq_len, batch_size, input_size]
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


def custom_collate_fn(batch):
    # Filter out any None samples
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None, None

    images, texts, _ = zip(*batch)

    # Define consistent transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.3490, 0.3219, 0.2957], std=[0.2993, 0.2850, 0.2735])
    ])


    max_width = max([img.shape[2] for img in images])
    processed_images = []
    for img in images:
        img = transforms.ToPILImage()(img.cpu())
        img = transform(img)
        padding = (0, max_width - img.shape[2], 0, 0)
        padded_img = F.pad(img, padding)
        processed_images.append(padded_img)

    # Stack images into a tensor of shape [batch_size, channels, height, width]
    images = torch.stack(processed_images, dim=0)


    return images, texts


def train(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    total_batches = 0
    for batch_idx, (images, texts) in enumerate(loader):
        if images is None or texts is None:
            continue  # Skip invalid samples

        images = images.to(device)

        # Encode texts
        encoded_texts, lengths = label_encoder.encode(texts)
        targets = torch.cat(encoded_texts).to(device)
        target_lengths = torch.tensor(lengths, dtype=torch.long).to(device)

        # Forward pass
        outputs = model(images)  # [T, N, C]
        outputs = F.log_softmax(outputs, dim=2)

        # Prepare input lengths
        batch_size = images.size(0)
        seq_len = outputs.size(0)
        input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long, device=device)

        # Compute loss
        loss = criterion(outputs, targets, input_lengths, target_lengths)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

        if batch_idx % 1 == 0:
            _, preds = outputs.max(2)
            preds = preds.transpose(1, 0).contiguous()  # [N, T]
            pred_texts = label_encoder.decode(preds)

            target_text = texts[0] if len(texts) > 0 else ''
            predicted_text = pred_texts[0] if len(pred_texts) > 0 else ''
            print(
                f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(loader)}], epoch loss: {running_loss / total_batches} , Loss: {loss.item():.4f}, "
                f"Target: '{target_text}', Predicted: '{predicted_text}'"
            )
            rotated_img = transforms.ToPILImage()(images[0].squeeze(0).clamp(0, 1))
            folder = "./backend/training_data/verify/text epoch " + str(epoch)
            if not os.path.exists(folder):
                os.makedirs(folder)
            rotated_img.save(folder + "/" + str(batch_idx) + ".png")

    epoch_loss = running_loss / total_batches
    return epoch_loss

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    total_batches = 0
    correct_chars = 0
    total_chars = 0

    with torch.no_grad():
        for images, texts in loader:
            if images is None or texts is None:
                continue  # Skip invalid samples

            images = images.to(device)

            # Encode texts
            encoded_texts, lengths = label_encoder.encode(texts)
            targets = torch.cat(encoded_texts).to(device)
            target_lengths = torch.tensor(lengths, dtype=torch.long).to(device)

            # Forward pass
            outputs = model(images)  # [T, N, C]
            outputs = F.log_softmax(outputs, dim=2)

            # Prepare input lengths
            batch_size = images.size(0)
            seq_len = outputs.size(0)
            input_lengths = torch.full(
                size=(batch_size,), fill_value=seq_len, dtype=torch.long
            ).to(device)

            # Compute CTC loss
            loss = criterion(outputs, targets, input_lengths, target_lengths)

            running_loss += loss.item()
            total_batches += 1

            # Decode predictions
            _, preds = outputs.max(2)
            preds = preds.transpose(1, 0).contiguous()  # [N, T]
            pred_texts = label_encoder.decode(preds)

            # Calculate character-level accuracy
            for pred_text, target_text in zip(pred_texts, texts):
                total_chars += len(target_text)
                correct_chars += sum(1 for p, t in zip(pred_text, target_text) if p == t)

    epoch_loss = running_loss / total_batches
    accuracy = (correct_chars / total_chars) * 100 if total_chars > 0 else 0.0
    return epoch_loss, accuracy


def verify_char_set(dataset, label_encoder):
    dataset_chars = set()
    for idx in range(len(dataset)):
        try:
            print(idx)
            _, text, _ = dataset[idx]
            dataset_chars.update(text)
        except:
            continue
    
    missing_chars = dataset_chars - set(label_encoder.char_set)
    if missing_chars:
        print(f"Missing characters in char_set: {missing_chars}")
        # Optionally, add missing characters to char_set and update label_encoder
    else:
        print("All dataset characters are present in char_set.")
        

class WrappedCTCLoss(nn.Module):
    def __init__(self, blank=0):
        super(WrappedCTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True)

    def forward(self, outputs, targets):
        encoded_texts, lengths = label_encoder.encode(targets)
        targets = torch.cat(encoded_texts).to(device)
        target_lengths = torch.tensor(lengths, dtype=torch.long).to(device)


        outputs = F.log_softmax(outputs, dim=2)

        # Prepare input lengths
        batch_size = outputs.size(1)
        seq_len = outputs.size(0)
        input_lengths = torch.full(
            size=(batch_size,), fill_value=seq_len, dtype=torch.long
        ).to(device)

        # Compute CTC loss
        loss = self.ctc_loss(outputs, targets, input_lengths, target_lengths)

        return loss

if __name__ == '__main__':
    batch_size = 64
    learning_rate = 4e-5
    weight_decay = 1e-4
    num_epochs = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Character set


    label_encoder = LabelEncoder(Constants.char_set)

    # Transforms
    transform = transforms.Compose([
        ResizeToMaxDimension(max_dim=Constants.desired_size),  # Resize based on max dimension while maintaining aspect ratio
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3490, 0.3219, 0.2957], std=[0.2993, 0.2850, 0.2735])
    ])

    # Datasets and DataLoaders
    train_dataset = CustomImageDataset2(img_dir='./backend/training_data/', transform=transform, train=True)
    test_dataset = CustomImageDataset2(img_dir='./backend/training_data/', transform=transform, train=False)


    #verify_char_set(train_dataset,label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,prefetch_factor=8,persistent_workers=True, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,prefetch_factor=2,persistent_workers=True, pin_memory=True,  collate_fn=custom_collate_fn)

    # Model, criterion, optimizer
    num_classes = len(Constants.char_set) + 1  # +1 for CTC blank label
    model = CRNN(num_classes=num_classes, nc=3).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # TensorBoard writer
    writer = SummaryWriter()

    if True:
        checkpoint = torch.load("bkCRNNmodel_checkpoint_5.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):  # Only move tensors
                        optimizer.state[param][key] = value.to(device)


    if False:
            # Initialize the learning rate finder with model, optimizer, and loss function
            lrCriterion = WrappedCTCLoss(blank=0).to(device)
            lr_finder = LRFinder(model, optimizer, lrCriterion, device=device)
            model.train()

            lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=0.01, num_iter=600)
            lr_finder.plot()

            plt.savefig('lr_finder_plot.png')  # Saves the plot to a file
            plt.show()  # Display the plot (optional)

            # Convert the loss list to a PyTorch Tensor
            losses_tensor = torch.tensor(lr_finder.history["loss"])

            # Find the index of the minimum loss
            min_loss_idx = torch.argmin(losses_tensor)

            # Retrieve the corresponding learning rate
            optimal_lr = lr_finder.history["lr"][min_loss_idx.item()]  # Use .item() to get Python scalar
            print(f"Optimal Learning Rate: {optimal_lr}")

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, epoch)
        #val_loss, val_acc = evaluate(model, test_loader, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')#, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        #writer.add_scalar('Loss/Validation', val_loss, epoch)
        #writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, f"CRNNmodel_checkpoint_{epoch+1}.pth")

    writer.close()

