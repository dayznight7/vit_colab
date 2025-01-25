from transformers import ViTForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch


'''
download imagenet-mini dataset here:
https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000

Folder
├── archive
│   └── imagenet-mini
│       ├── train
│       └── val
│           └── #class
│               └── .jpg
└── vit_accuracy.py

env:
transformers=4.45.2
torch=2.5.1+cu118
@cuDNN=9.1.0
@python=3.10.11
@OS=win11
'''


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

imagenet_data = datasets.ImageFolder(root='archive/imagenet-mini/val', transform=transform)
dataloader_imagenet_mini = DataLoader(imagenet_data, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataloader_imagenet_mini:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs.logits, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
# Accuracy: 77.42%
