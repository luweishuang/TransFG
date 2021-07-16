import numpy as np
import cv2
import time
import os
import argparse
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from PIL import Image
from torchvision import transforms
from models.modeling import VisionTransformer, CONFIGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["CUB_200_2011", "emptyJudge5", "emptyJudge4"], default="emptyJudge5",
                        help="Which dataset.")
    parser.add_argument("--img_size", default=448, type=int, help="Resolution size")
    parser.add_argument('--split', type=str, default='overlap', help="Split method")  # non-overlap
    parser.add_argument('--slide_step', type=int, default=12, help="Slide step for overlap split")
    parser.add_argument('--smoothing_value', type=float, default=0.0, help="Label smoothing value\n")
    parser.add_argument("--pretrained_model", type=str, default="output/emptyjudge5_checkpoint.bin",
                        help="load pretrained model")
    return parser.parse_args()


class Predictor(object):
    def __init__(self, args):
        self.args = args
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.nprocs = torch.cuda.device_count()

        self.cls_dict = {}
        self.num_classes = 0
        self.model = None
        self.prepare_model()
        self.test_transform = transforms.Compose([transforms.Resize((448, 448), Image.BILINEAR),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def prepare_model(self):
        config = CONFIGS["ViT-B_16"]
        config.split = self.args.split
        config.slide_step = self.args.slide_step

        if self.args.dataset == "emptyJudge5":
            self.num_classes = 5
            self.cls_dict = {0: "noemp", 1: "yesemp", 2: "hard", 3: "fly", 4: "stack"}
        elif self.args.dataset == "emptyJudge4":
            self.num_classes = 4
            self.cls_dict = {0: "noemp", 1: "yesemp", 2: "hard", 3: "stack"}
        elif self.args.dataset == "emptyJudge3":
            self.num_classes = 3
            self.cls_dict = {0: "noemp", 1: "yesemp", 2: "hard"}
        elif self.args.dataset == "emptyJudge2":
            self.num_classes = 2
            self.cls_dict = {0: "noemp", 1: "yesemp"}
        self.model = VisionTransformer(config, self.args.img_size, zero_head=True, num_classes=self.num_classes, smoothing_value=self.args.smoothing_value)
        if self.args.pretrained_model is not None:
            pretrained_model = torch.load(self.args.pretrained_model, map_location=torch.device('cpu'))['model']
            self.model.load_state_dict(pretrained_model)
        self.model.to(self.args.device)
        self.model.eval()

    def normal_predict(self, img_path):
        # img = cv2.imread(img_path)
        img = Image.open(img_path)
        if img is None:
            print(
                "Image file failed to read: {}".format(img_path))
        else:
            x = self.test_transform(img)
            part_logits = self.model(x.unsqueeze(0))
            probs = torch.nn.Softmax(dim=-1)(part_logits)
            topN = torch.argsort(probs, dim=-1, descending=True).tolist()
            clas_ids = topN[0][0]
            # print(probs[0, topN[0][0]].item())
            return clas_ids, probs[0, clas_ids].item()


if __name__ == "__main__":
    args = parse_args()
    predictor = Predictor(args)

    y_true = []
    y_pred = []
    test_dir = "/data/fineGrained/test_imgs"
    dir_dict = {"noemp":"0", "yesemp":"1", "hard": "2", "fly": "3", "stack": "4"}
    total = 0
    num = 0
    t0 = time.time()
    for dir_name, label in dir_dict.items():
        cur_folder = os.path.join(test_dir, dir_name)
        errorPath = os.path.join(test_dir, dir_name + "_error")
        # os.makedirs(errorPath, exist_ok=True)
        for cur_file in os.listdir(cur_folder):
            total += 1
            print("%d processing: %s" % (total, cur_file))
            cur_img_file = os.path.join(cur_folder, cur_file)
            error_img_dst = os.path.join(errorPath, cur_file)
            cur_pred, pred_score = predictor.normal_predict(cur_img_file)

            label = 0 if 2 == int(label) or 3 == int(label) or 4 == int(label) else int(label)
            cur_pred = 0 if 2 == int(cur_pred) or 3 == int(cur_pred) or 4 == int(cur_pred) else int(cur_pred)
            y_true.append(int(label))
            y_pred.append(int(cur_pred))
            if int(label) == int(cur_pred):
                num += 1
            else:
                print(cur_file, "predict: ", cur_pred, "true: ", int(label))
                print(cur_file, "predict: ", cur_pred, "true: ", int(label), "pred_score:", pred_score)
                os.system("cp %s %s" % (cur_img_file, error_img_dst))
    t1 = time.time()
    print('The cast of time is :%f seconds' % (t1-t0))
    rate = float(num)/total
    print('The classification accuracy is %f' % rate)

    rst_C = confusion_matrix(y_true, y_pred)
    rst_f1 = f1_score(y_true, y_pred, average='macro')
    print(rst_C)
    print(rst_f1)

'''

'''
