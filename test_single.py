# coding=utf-8
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
from models.modeling import VisionTransformer, CONFIGS


parser = argparse.ArgumentParser()
parser.add_argument("--img_size", default=448, type=int, help="Resolution size")
parser.add_argument('--split', type=str, default='overlap', help="Split method")  # non-overlap
parser.add_argument('--slide_step', type=int, default=12, help="Slide step for overlap split")
parser.add_argument('--smoothing_value', type=float, default=0.0, help="Label smoothing value\n")
parser.add_argument("--pretrained_model", type=str, default="output/emptyjudge4_checkpoint.bin", help="load pretrained model")
args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.nprocs = torch.cuda.device_count()

# Prepare Model
config = CONFIGS["ViT-B_16"]
config.split = args.split
config.slide_step = args.slide_step

cls_dict = {}
num_classes = 0
model_name = os.path.basename(args.pretrained_model).replace("_checkpoint.bin", "")
print("use model_name: ", model_name)
if model_name.lower() == "emptyJudge5".lower():
    num_classes = 5
    cls_dict = {0: "noemp", 1: "yesemp", 2: "hard", 3: "fly", 4: "stack"}
elif model_name.lower() == "emptyJudge4".lower():
    num_classes = 4
    cls_dict = {0: "noemp", 1: "yesemp", 2: "hard", 3: "stack"}
elif model_name.lower() == "emptyJudge3".lower():
    num_classes = 3
    cls_dict = {0: "noemp", 1: "yesemp", 2: "hard"}
elif model_name.lower() == "emptyJudge2".lower():
    num_classes = 2
    cls_dict = {0: "noemp", 1: "yesemp"}
model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)
if args.pretrained_model is not None:
    pretrained_model = torch.load(args.pretrained_model, map_location=torch.device('cpu'))['model']
    model.load_state_dict(pretrained_model)
model.to(args.device)
model.eval()
# test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
#                             transforms.CenterCrop((448, 448)),
#                             transforms.ToTensor(),
#                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize((448, 448), Image.BILINEAR),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
img = Image.open("ckpts/img.jpg")
x = test_transform(img)
part_logits = model(x.unsqueeze(0))

probs = torch.nn.Softmax(dim=-1)(part_logits)
top5 = torch.argsort(probs, dim=-1, descending=True)
print("Prediction Label\n")
for idx in top5[0, :5]:
    print(f'{probs[0, idx.item()]:.5f} : {cls_dict[idx.item()]}', end='\n')

