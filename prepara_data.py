import os
import cv2
import numpy as np
import subprocess
import random


# ----------- 改写名称 --------------
# index = 0
# src_dir = "/data/fineGrained/emptyJudge5"
# dst_dir = src_dir + "_new"
# os.makedirs(dst_dir, exist_ok=True)
# for sub in os.listdir(src_dir):
#     sub_path = os.path.join(src_dir, sub)
#     sub_path_dst = os.path.join(dst_dir, sub)
#     os.makedirs(sub_path_dst, exist_ok=True)
#     for cur_f in os.listdir(sub_path):
#         cur_img = os.path.join(sub_path, cur_f)
#         cur_img_dst = os.path.join(sub_path_dst, "a%05d.jpg" % index)
#         index += 1
#         os.system("mv %s %s" % (cur_img, cur_img_dst))


# ----------- 删除过小图像 --------------
# src_dir = "/data/fineGrained/emptyJudge5"
# for sub in os.listdir(src_dir):
#     sub_path = os.path.join(src_dir, sub)
#     for cur_f in os.listdir(sub_path):
#         filepath = os.path.join(sub_path, cur_f)
#         res = subprocess.check_output(['file', filepath])
#         pp = res.decode("utf-8").split(",")[-2]
#         height = int(pp.split("x")[1])
#         width = int(pp.split("x")[0])
#         min_l = min(height, width)
#         if min_l <= 448:
#             os.system("rm %s" % filepath)


# ----------- 获取有效图片并写images.txt --------------
# src_dir = "/data/fineGrained/emptyJudge4/images"
# src_dict = {"noemp":"0", "yesemp":"1", "hard": "2", "stack": "3"}
# all_dict = {"yesemp":[], "noemp":[], "hard": [], "stack": []}
# for sub, value in src_dict.items():
#     sub_path = os.path.join(src_dir, sub)
#     for cur_f in os.listdir(sub_path):
#         all_dict[sub].append(os.path.join(sub, cur_f))
#
# yesnum = len(all_dict["yesemp"])
# nonum = len(all_dict["noemp"])
# hardnum = len(all_dict["hard"])
# stacknum = len(all_dict["stack"])
# thnum = min(yesnum, nonum, hardnum, stacknum)
# images_txt = src_dir + ".txt"
# index = 1
#
# def write_images(cur_list, thnum, fw, index):
#     for feat_path in random.sample(cur_list, thnum):
#         fw.write(str(index) + " " + feat_path + "\n")
#         index += 1
#     return index
#
# with open(images_txt, "w") as fw:
#     index = write_images(all_dict["noemp"], thnum, fw, index)
#     index = write_images(all_dict["yesemp"], thnum, fw, index)
#     index = write_images(all_dict["hard"], thnum, fw, index)
#     index = write_images(all_dict["stack"], thnum, fw, index)

# ----------- 写 image_class_labels.txt + train_test_split.txt --------------
src_dir = "/data/fineGrained/emptyJudge4"
src_dict = {"noemp":"0", "yesemp":"1", "hard": "2", "stack": "3"}
images_txt = os.path.join(src_dir, "images.txt")
image_class_labels_txt = os.path.join(src_dir, "image_class_labels.txt")
imgs_cnt = 0
with open(image_class_labels_txt, "w") as fw:
    with open(images_txt, "r") as fr:
        for cur_l in fr:
            imgs_cnt += 1
            img_index, img_f = cur_l.strip().split(" ")
            folder_name = img_f.split("/")[0]
            if folder_name in src_dict:
                cur_line = img_index + " " + str(int(src_dict[folder_name])+1)
                fw.write(cur_line + "\n")

train_num = int(imgs_cnt*0.85)
print("train_num= ", train_num, ", imgs_cnt= ", imgs_cnt)
all_list = [1]*train_num + [0]*(imgs_cnt-train_num)
assert len(all_list) == imgs_cnt
random.shuffle(all_list)
train_test_split_txt = os.path.join(src_dir, "train_test_split.txt")
with open(train_test_split_txt, "w") as fw:
    with open(images_txt, "r") as fr:
        for cur_l in fr:
            img_index, img_f = cur_l.strip().split(" ")
            cur_line = img_index + " " + str(all_list[int(img_index) - 1])
            fw.write(cur_line + "\n")
