# -*- coding: UTF-8 -*-
import cv2
import os
from tqdm import tqdm

path = '/media/qzj/Document/grow/research/slamDataSet/SJTU/PR/round4'
for i in tqdm(range(12)):
    sub_dir = str(i).zfill(2)
    root = os.path.join(path,sub_dir)

    # 复制到25张
    # files = sorted(os.listdir(root))
    # copy_i = 0
    # while(len(os.listdir(root))<25):
    #     file_name = files[copy_i%len(files)]
    #     img = cv2.imread(os.path.join(root,file_name))
    #     file_name_new = file_name.replace(".jpg", "{}.jpg".format(copy_i))
    #     cv2.imwrite(os.path.join(root,file_name_new),img)
    #     copy_i = copy_i + 1

    for file in os.listdir(root):
        if(file.endswith(".jpg")):
            img = cv2.imread(os.path.join(root,file))
            img_resize = cv2.resize(img, (1024, 768))
            cv2.imwrite(os.path.join(root,file),img_resize)
