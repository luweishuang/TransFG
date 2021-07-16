# coding=utf-8
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
from models.modeling import VisionTransformer, CONFIGS
import scipy.misc


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/data/fineGrained')
parser.add_argument("--dataset",
                    choices=["CUB_200_2011", "emptyJudge5", "emptyJudge4"], default="emptyJudge5", help="Which dataset.")
parser.add_argument("--img_size", default=448, type=int, help="Resolution size")
parser.add_argument('--split', type=str, default='overlap', help="Split method")  # non-overlap
parser.add_argument('--slide_step', type=int, default=12, help="Slide step for overlap split")
parser.add_argument('--smoothing_value', type=float, default=0.0, help="Label smoothing value\n")
parser.add_argument("--pretrained_model", type=str, default="output/emptyjudge5_checkpoint.bin", help="load pretrained model")
args = parser.parse_args()
args.data_root = '{}/{}'.format(args.data_root, args.dataset)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.nprocs = torch.cuda.device_count()

os.makedirs("attention_data", exist_ok=True)
cls_dict = {0: "noemp", 1: "yesemp", 2: "hard", 3: "fly", 4: "stack"}
imagenet_labels = dict(enumerate(open('attention_data/ilsvrc2012_wordnet_lemmas.txt')))

# Prepare Model
config = CONFIGS["ViT-B_16"]
config.split = args.split
config.slide_step = args.slide_step

if args.dataset == "emptyJudge5":
    num_classes = 5
elif args.dataset == "emptyJudge4":
    num_classes = 4
elif args.dataset == "emptyJudge3":
    num_classes = 3
model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)
model.load_from(np.load("attention_data/ViT-B_16-224.npz"))
if args.pretrained_model is not None:
    pretrained_model = torch.load(args.pretrained_model, map_location=torch.device('cpu'))['model']
    model.load_state_dict(pretrained_model)
model.to(args.device)
model.eval()

test_transform = transforms.Compose([transforms.Resize((448, 448), Image.BILINEAR),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
img = scipy.misc.imread("attention_data/img.jpg")
if len(img.shape) == 2:
    img = np.stack([img] * 3, 2)
im11 = Image.fromarray(img, mode='RGB')

im22 = Image.open("attention_data/img.jpg")
x = test_transform(im22)
part_logits = model(x.unsqueeze(0))

probs = torch.nn.Softmax(dim=-1)(part_logits)
top5 = torch.argsort(probs, dim=-1, descending=True)
print("Prediction Label\n")
for idx in top5[0, :5]:
    print(f'{probs[0, idx.item()]:.5f} : {cls_dict[idx.item()]}', end='\n')

