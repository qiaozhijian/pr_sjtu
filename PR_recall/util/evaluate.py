import torch
import torch.nn as nn
from util.sjtu import load_obj,save_obj
import os
from os.path import join
import numpy as np
from model.netvlad import NetVLAD
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import threading
import matplotlib.pyplot as plt
from options.test_options import TestOptions
from model.disam_model import DISAM_Model

encoder_dim = 512
pretrained = True
num_clusters = 64
vladv2 = False
pool_size = encoder_dim * num_clusters


def getDISAM():
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.use_two_stage = True
    opt.isTrain = False
    opt.which_epoch = 1200
    opt.which_epoch_finer = 1200
    opt.checkpoints_dir = './DISAM'
    opt.name = 'CMU_coarse'
    opt.name_finer = 'CMU_fine'
    opt.n_domains = 12

    model = DISAM_Model(opt)

    corse_model = model.netG.encode
    finer_model = model.netG_finer.encode

    return corse_model, finer_model

def plotFig(recall, precision, name = ''):

    fig, ax = plt.subplots()
    ax.plot(recall, precision)

    ax.set(xlabel='Recall (%)', ylabel='Precision (%)',
           title='Precision-Recall')
    ax.grid()
    # 设置坐标轴范围
    # plt.xlim((0, 1))
    # plt.ylim((0, 1))

    if name != '':
        fig.savefig(name+".png")
    else:
        plt.show()
    plt.close()

class RecallN (threading.Thread):
    def __init__(self, feat1, feat2, n):
        threading.Thread.__init__(self)
        self.feat1 = feat1
        self.feat2 = feat2
        self.n = n
        self.recall_n = 0
    def run(self):
        totol_n = 0
        # (n,128) (1,128)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.feat2)
        recall_n = 0
        for i in range(self.feat1.shape[0]):
            distances, positives = knn.kneighbors(self.feat1[i].reshape(1,-1), n_neighbors=self.n)
            true_positive = np.arange(i%25*25,(i%25+1)*25)
            re = np.intersect1d(positives,true_positive)
            if re.size>0:
                recall_n = recall_n + 1
            totol_n = totol_n + 1
        recall_n = recall_n / totol_n
        self.recall_n = recall_n

def recall_precision_n(feaVec1, feaVec2, recall_num = 25):
    feat1 = feaVec1['feaVec']
    feat2 = feaVec2['feaVec']

    recall_list = np.zeros(shape=recall_num)
    precision_list = np.zeros(shape=recall_num)
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(feat2)

    totol_n = 0
    for i in tqdm(range(feat1.shape[0])):
        distances, positives = knn.kneighbors(feat1[i].reshape(1, -1), n_neighbors=recall_num)
        positives = positives[0]
        true_positive = np.arange(i // 25 * 25, (i // 25 + 1) * 25)
        totol_n = totol_n + 1
        # print(i,positives[0])
        for n in range(recall_num):
            re = np.intersect1d(positives[:n+1], true_positive)
            precision_list[n] = precision_list[n] + re.size / (n + 1)
            if re.size > 0:
                recall_list[n] = recall_list[n] + 1
    recall_list = recall_list / totol_n
    precision_list = precision_list / totol_n

    return recall_list, precision_list

def cos_dis(f1,f2):

    mean_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

    return mean_cos(f1.view(256, -1), f2.view(256, -1)).mean(0)

def recall_precision_n_DISAM(feat1, feat2, recall_num = 25):

    f1 = feat1.cuda()
    f2 = feat2.cuda()
    dis_nn = torch.empty(size=(300,300))
    mean_cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
    for idx_query in tqdm(range(300)) :
        for idx_data in range(300):
            dis_tmp = mean_cos(f1[idx_query], f2[idx_data]).mean(dim=0)
            dis_nn[idx_query, idx_data] = dis_tmp

    values,indices = torch.topk(dis_nn,dim=1,largest=True,sorted=True,k=recall_num*2)

    recall_list = np.zeros(shape=recall_num)
    precision_list = np.zeros(shape=recall_num)
    totol_n = 0
    for i in tqdm(range(feat1.shape[0])):
        positives = indices[i,:recall_num].detach().cpu().numpy()
        true_positive = np.arange(i // 25 * 25, (i // 25 + 1) * 25)
        totol_n = totol_n + 1
        # print(i,positives[0])
        for n in range(recall_num):
            re = np.intersect1d(positives[:n+1], true_positive)
            precision_list[n] = precision_list[n] + re.size / (n + 1)
            if re.size > 0:
                recall_list[n] = recall_list[n] + 1
    recall_list = recall_list / totol_n
    precision_list = precision_list / totol_n

    return recall_list, precision_list

def recall_precision_direct(dis_nn,recall_num=25):
    values, indices = torch.topk(dis_nn, dim=1, largest=True, sorted=True, k=recall_num * 2)

    recall_list = np.zeros(shape=recall_num)
    precision_list = np.zeros(shape=recall_num)
    totol_n = 0
    for i in (range(300)):
        positives = indices[i, :recall_num].detach().cpu().numpy()
        true_positive = np.arange(i // 25 * 25, (i // 25 + 1) * 25)
        totol_n = totol_n + 1
        # print(i,positives[0])
        for n in range(recall_num):
            re = np.intersect1d(positives[:n + 1], true_positive)
            precision_list[n] = precision_list[n] + re.size / (n + 1)
            if re.size > 0:
                recall_list[n] = recall_list[n] + 1
    recall_list = recall_list / totol_n
    precision_list = precision_list / totol_n

    return recall_list, precision_list

def recall_precision_n_DISAM_corse_fine(feat1_c, feat2_c, feat1_f, feat2_f, recall_num = 25):

    dis_nn_c = torch.empty(size=(300,300))
    dis_nn_f = torch.empty(size=(300,300))
    mean_cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
    for idx_query in tqdm(range(300)) :
        for idx_data in range(300):
            dis_tmp = mean_cos(feat1_c[idx_query].cuda(), feat2_c[idx_data].cuda()).mean(dim=0)
            dis_nn_c[idx_query, idx_data] = dis_tmp
            dis_tmp = mean_cos(feat1_f[idx_query].cuda(), feat2_f[idx_data].cuda()).mean(dim=0)
            dis_nn_f[idx_query, idx_data] = dis_tmp

    recall_list_c, precision_list_c = recall_precision_direct(dis_nn_c,recall_num=25)
    recall_list_f, precision_list_f = recall_precision_direct(dis_nn_f,recall_num=25)

    values_c,indices_c = torch.topk(dis_nn_c,dim=1,largest=True,sorted=True,k=recall_num*2)
    values_f,indices_f = torch.topk(dis_nn_f,dim=1,largest=True,sorted=True,k=300)
    indice = np.zeros(shape=(300,recall_num))
    for i in range(300):
        num = 0
        idx = 0
        while(num<recall_num):
            if indices_f[i,idx] in indices_c[i]:
                indice[i,num] = indices_f[i,idx]
                num = num+1
            idx = idx+1

    recall_list = np.zeros(shape=recall_num)
    precision_list = np.zeros(shape=recall_num)
    totol_n = 0
    for i in (range(300)):
        positives = indice[i,:recall_num]
        true_positive = np.arange(i // 25 * 25, (i // 25 + 1) * 25)
        totol_n = totol_n + 1
        # print(i,positives[0])
        for n in range(recall_num):
            re = np.intersect1d(positives[:n+1], true_positive)
            precision_list[n] = precision_list[n] + re.size / (n + 1)
            if re.size > 0:
                recall_list[n] = recall_list[n] + 1
    recall_list_cf = recall_list / totol_n
    precision_list_cf = precision_list / totol_n

    return recall_list_c, precision_list_c, recall_list_f, precision_list_f, recall_list_cf, precision_list_cf

def getNetVLAD():

    encoder = models.vgg16(pretrained=pretrained)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]

    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder', encoder)

    net_vlad = NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=vladv2)

    model.add_module('pool', net_vlad)

    resume_ckpt = join('./vgg16_netvlad_checkpoint', 'checkpoints', 'checkpoint.pth.tar')
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    return model

def getFeaVec_NetVLAD(model, imgs, pool_size, seq = 1):

    if os.path.exists('obj/NetVLAD_{}.pkl'.format(seq)):
        print('load feature {} from obj'.format(seq))
        return load_obj('NetVLAD_{}'.format(seq))

    feaVecs = {}
    model.eval()
    total_n = 0
    for i in range(12):
        total_n = total_n + len(imgs[i])
    feaVecs['feaVec'] = np.empty((total_n, pool_size))
    feaVecs['indice'] = []
    with torch.no_grad():
        cnt = 0
        for i in tqdm(range(12)):
            sub_imgs = imgs[i]
            n = len(sub_imgs)
            for j in range(n):
                input = imgs[i][j].cuda().unsqueeze(0)
                image_encoding = model.encoder(input)
                vlad_encoding = model.pool(image_encoding)
                feaVecs['feaVec'][cnt,:] = vlad_encoding.detach().cpu().numpy()
                feaVecs['indice'].append(i*100+j)
                cnt = cnt + 1

    save_obj(feaVecs, 'NetVLAD_{}'.format(seq))
    print('save feature {} from obj'.format(seq))

    return feaVecs


def getFeaVec_DISAM(model, imgs, seq = 1, mode = 'corse',domain=0):

    if not os.path.exists('obj'):
        os.mkdir('obj')

    if os.path.exists('obj/DISAM_domain{}_{}_{}.pkl'.format(domain, mode, seq)):
        print('load {} feature {} from obj'.format(mode,seq))
        return load_obj('DISAM_domain{}_{}_{}'.format(domain, mode,seq))

    total_n = 0
    for i in range(12):
        total_n = total_n + len(imgs[i])
    feaVecs = []
    with torch.no_grad():
        cnt = 0
        for i in tqdm(range(12)):
            sub_imgs = imgs[i]
            n = len(sub_imgs)
            for j in range(n):
                input = imgs[i][j].cuda().unsqueeze(0)
                image_encoding = model(input,domain)
                feaVecs.append(image_encoding.detach().cpu())
                cnt = cnt + 1

    size_tmp = len(feaVecs)
    feaVecs_t = torch.empty(size=(size_tmp,256,64*64))
    for i in range(size_tmp):
        feaVecs_t[i]= feaVecs[i].view(256,-1)
    feaVecs_t = feaVecs_t.detach().cpu()
    save_obj(feaVecs_t, 'DISAM_domain{}_{}_{}'.format(domain, mode,seq))
    print('save {} feature {} from obj'.format(mode, seq))

    return feaVecs_t