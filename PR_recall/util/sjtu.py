import os
import torchvision.transforms as transforms
import cv2
from PIL import Image
from tqdm import tqdm


import pickle
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def input_transform_NetVLAD():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def input_transform_DISAM():
    osize = [256, 256]
    return transforms.Compose([
        transforms.Resize(osize, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

def sjtu_seq_NetVLAD(seq = 1):
    if seq < 1 or seq > 4:
        OSError('seq < 1 or seq > 4')

    if os.path.exists('obj/imgs_netvlad_{}.pkl'.format(seq)):
        print('load imgs {} from obj'.format(seq))
        return load_obj('imgs_netvlad_{}'.format(seq))

    path = '/media/qzj/Document/grow/research/slamDataSet/SJTU/PR/round{}'.format(seq)
    imgs={}
    for i in tqdm(range(12)):
        sub_dir = str(i).zfill(2)
        root = os.path.join(path, sub_dir)
        files = sorted(os.listdir(root))
        imgs[i] = []
        for file in files:
            if (file.endswith(".jpg")):
                img = Image.open(os.path.join(root, file))
                img = input_transform_NetVLAD()(img)
                imgs[i].append(img)

    save_obj(imgs, 'imgs_netvlad_{}'.format(seq))
    print('save imgs {} from obj'.format(seq))

    return imgs

def sjtu_seq_DISAM(seq = 1):
    if seq < 1 or seq > 4:
        OSError('seq < 1 or seq > 4')

    if os.path.exists('obj/imgs_disam_{}.pkl'.format(seq)):
        print('load imgs {} from obj'.format(seq))
        return load_obj('imgs_disam_{}'.format(seq))

    path = '/media/qzj/Document/grow/research/slamDataSet/SJTU/PR/round{}'.format(seq)
    imgs={}
    for i in tqdm(range(12)):
        sub_dir = str(i).zfill(2)
        root = os.path.join(path, sub_dir)
        files = sorted(os.listdir(root))
        imgs[i] = []
        for file in files:
            if (file.endswith(".jpg")):
                img = Image.open(os.path.join(root, file))
                img = input_transform_DISAM()(img)
                imgs[i].append(img)

    save_obj(imgs, 'imgs_disam_{}'.format(seq))
    print('save imgs {} from obj'.format(seq))

    return imgs