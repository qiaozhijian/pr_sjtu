from util.sjtu import sjtu_seq_DISAM
from util.evaluate import getDISAM,getFeaVec_DISAM,recall_precision_n_DISAM, plotFig,recall_precision_n_DISAM_corse_fine
import torch
import numpy as np
from tqdm import tqdm
from os.path import exists
from os import mkdir
import argparse
import sys

def createDir(domain_corse=0,domain_fine=1):
    path = './figure/DISAM/domain{}_{}/'.format(domain_corse,domain_fine)
    if not exists('figure'):
        mkdir('figure')
    if not exists('figure/DISAM'):
        mkdir('figure/DISAM')
    if not exists(path):
        mkdir(path)
    if not exists(path+'corse_fine'):
        mkdir(path+'corse_fine')
    if not exists(path+'fine'):
        mkdir(path+'fine')
    if not exists(path+'corse'):
        mkdir(path+'corse')
    return path


def test(root, domain_corse=0, domain_fine=0):
    corse_model, finer_model = getDISAM()
    print('load image')

    img_r1 = sjtu_seq_DISAM(root, seq=1)
    img_r2 = sjtu_seq_DISAM(root, seq=2)
    img_r3 = sjtu_seq_DISAM(root, seq=3)
    img_r4 = sjtu_seq_DISAM(root, seq=4)

    print('\nmodel inference')
    feaVecs_r1_corse = getFeaVec_DISAM(corse_model, img_r1, seq=1, mode='corse',domain=domain_corse)

    # recall_list_c, precision_list_c, recall_list_f, precision_list_f, recall_list_cf, precision_list_cf = \
    #     recall_precision_n_DISAM_corse_fine(feaVecs_r1_corse, feaVecs_r1_corse, feaVecs_r1_corse, feaVecs_r1_corse,
    #                                         recall_num=25)

    feaVecs_r2_corse = getFeaVec_DISAM(corse_model, img_r2, seq=2, mode='corse',domain=domain_corse)
    feaVecs_r3_corse = getFeaVec_DISAM(corse_model, img_r3, seq=3, mode='corse',domain=domain_corse)
    feaVecs_r4_corse = getFeaVec_DISAM(corse_model, img_r4, seq=4, mode='corse',domain=domain_corse)

    feaVecs_r1_finer = getFeaVec_DISAM(finer_model, img_r1, seq=1, mode='finer',domain=domain_fine)
    feaVecs_r2_finer = getFeaVec_DISAM(finer_model, img_r2, seq=2, mode='finer',domain=domain_fine)
    feaVecs_r3_finer = getFeaVec_DISAM(finer_model, img_r3, seq=3, mode='finer',domain=domain_fine)
    feaVecs_r4_finer = getFeaVec_DISAM(finer_model, img_r4, seq=4, mode='finer',domain=domain_fine)

    feaVecs_corse = [feaVecs_r1_corse, feaVecs_r2_corse, feaVecs_r3_corse, feaVecs_r4_corse]
    feaVecs_finer = [feaVecs_r1_finer, feaVecs_r2_finer, feaVecs_r3_finer, feaVecs_r4_finer]

    path = createDir(domain_corse=domain_corse,domain_fine=domain_fine)
    f = open(path + 'DISAM_domain_corse_{}_fine_{}.txt'.format(domain_corse, domain_fine),mode='w')
    for i in range(4):
        for j in range(4):
            if i==j: continue
            if i==1 or j==1: continue
            recall_list_c, precision_list_c, recall_list_f, precision_list_f, recall_list_cf, precision_list_cf = \
                recall_precision_n_DISAM_corse_fine(feaVecs_corse[i], feaVecs_corse[j], feaVecs_finer[i], feaVecs_finer[j], recall_num = 25)

            f.write("{}_{}_recall_c:\n".format(i,j))
            f.write(str(recall_list_c)+'\n')
            f.write("{}_{}_precision_c:\n".format(i,j))
            f.write(str(precision_list_c)+'\n')
            f.write("{}_{}_recall_f:\n".format(i,j))
            f.write(str(recall_list_f)+'\n')
            f.write("{}_{}_precision_f:\n".format(i,j))
            f.write(str(precision_list_f)+'\n')
            f.write("{}_{}_recall_cf:\n".format(i,j))
            f.write(str(recall_list_cf)+'\n')
            f.write("{}_{}_precision_cf:\n".format(i,j))
            f.write(str(precision_list_cf)+'\n')

            plotFig(recall_list_cf, precision_list_cf, name=path + "corse_fine/{}_{}".format(i,j))
            plotFig(recall_list_f, precision_list_f, name=path + "fine/{}_{}".format(i,j))
            plotFig(recall_list_c, precision_list_c, name=path + "corse/{}_{}".format(i,j))
            print('finish {}_{}'.format(i,j))
    f.close()

if __name__ == "__main__":

    root = '../../dataset/SJTU/PR/'
    for i in range(2,9):
        for j in range(2,9):
            test(root, domain_corse=i, domain_fine=j)
