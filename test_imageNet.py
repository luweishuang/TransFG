# coding=utf-8
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.modeling import VisionTransformer, CONFIGS

os.makedirs("attention_data", exist_ok=True)
imagenet_labels = dict(enumerate(open('attention_data/ilsvrc2012_wordnet_lemmas.txt')))

# Prepare Model
config = CONFIGS["ViT-B_16"]
model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224)
model.load_from(np.load("attention_data/ViT-B_16-224.npz"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
im = Image.open("attention_data/img.jpg")
x = transform(im)
print("x.size() = ", x.size())

part_logits = model(x.unsqueeze(0))

probs = torch.nn.Softmax(dim=-1)(part_logits)
top5 = torch.argsort(probs, dim=-1, descending=True)
print("Prediction Label\n")
for idx in top5[0, :5]:
    print(f'{probs[0, idx.item()]:.5f} : {imagenet_labels[idx.item()]}', end='')

