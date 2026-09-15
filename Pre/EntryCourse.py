import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image, ImageFilter
import json

DEVICE = "cuda"

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

classes = load_json('data/imagenet-simple-labels.json') 

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open("data/guitar.jpg").convert('RGB')
blurred_image = image.filter(ImageFilter.GaussianBlur(radius=0))
image_tensor = transform(blurred_image).unsqueeze(0)
image_tensor = image_tensor.to(DEVICE)

with torch.no_grad():
    outputs = model(image_tensor)

    max = torch.argmax(outputs[0], dim=0)
    print(f"Уверенный результат по индексу: {max}, Класс: {classes[max]}")

    probs = torch.softmax(outputs[0], dim=0)
    top5_probs, top5_indices = torch.topk(probs, k=5)

    print(f"Топ 5 вероятностей: {top5_probs}")
    print(f"Топ 5 индексов: {top5_indices}\n")

    # Выводим результат
    for i, (idx, prob) in enumerate(zip(top5_indices.tolist(), top5_probs.tolist())):
        print(f"{i+1}. Класс: {classes[idx]}, Индекс {idx}, Вероятность: {prob:.2%}")