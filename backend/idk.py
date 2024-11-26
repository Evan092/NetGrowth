import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO



def load_and_preprocess_image(url):
    preprocess = transforms.Compose([
        transforms.Resize(256), # Resize the shortest side to 256 pixels
        transforms.CenterCrop(224), # Crop the center 224x224 pixels
        transforms.ToTensor(), # Convert PIL Image to tensor
        transforms.Normalize( # Normalize as per ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        ),
    ])
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def load_imagenet_labels():
    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels_response = requests.get(LABELS_URL)
    labels = labels_response.json()
    return labels

def get_top5_predictions(probabilities, labels):
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    top5 = []
    for i in range(top5_prob.size(1)):
        label = labels[top5_catid[0][i]]
        prob = top5_prob[0][i].item() * 100
        top5.append((label, prob))
    return top5


image_url = "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01667778_terrapin.JPEG?raw=true"
input_batch = load_and_preprocess_image(image_url)
labels = load_imagenet_labels()

model = models.vit_b_16(pretrained=True)
model.eval()

with torch.no_grad():
    vit_outputs = model(input_batch)
    vit_logits = vit_outputs
    vit_probabilities = torch.nn.functional.softmax(vit_logits, dim=1)
    vit_top5 = get_top5_predictions(vit_probabilities,labels)


resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()
with torch.no_grad():
    resnet_outputs = resnet_model(input_batch)
    resnet_probabilities = torch.nn.functional.softmax(resnet_outputs, dim=1)
    resnet_top5 = get_top5_predictions(resnet_probabilities, labels)


print("----- Vision Transformer (ViT) Top 5 Predictions -----")
for label, prob in vit_top5:
    print(f"{label}: {prob:.2f}%")
    print("\n----- ResNet-50 Top 5 Predictions -----")
    for label, prob in resnet_top5:
        print(f"{label}: {prob:.2f}%")
