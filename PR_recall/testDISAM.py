from util.sjtu import sjtu_seq_DISAM
from util.evaluate import getDISAM,getFeaVec_DISAM,recall_precision_n_DISAM, plotFig,recall_precision_n_DISAM_corse_fine
import torch
import numpy as np
from tqdm import tqdm
from os.path import exists
from os import mkdir
import argparse
import sys

def createDir(query, data, domain_query=0,domain_data=1):
    path = './figure/DISAM/scene{}_{}/'.format(query,data)
    if not exists('figure'):
        mkdir('figure')
    if not exists('figure/DISAM'):
        mkdir('figure/DISAM')
    if not exists(path):
        mkdir(path)
    path = path + 'domain{}_{}/'.format(domain_query,domain_data)
    if not exists(path):
        mkdir(path)
    if not exists(path+'corse_fine/'):
        mkdir(path+'corse_fine/')
    if not exists(path+'fine/'):
        mkdir(path+'fine/')
    if not exists(path+'corse/'):
        mkdir(path+'corse/')
    return path

def test(root, corse_model, finer_model, query, data, domain_query=0, domain_data=0):
    if query == data:
        return 0
    if query == 2 or data == 2:
        return 0
    print('load image')
    query_imgs = sjtu_seq_DISAM(root, seq=query)
    data_imgs = sjtu_seq_DISAM(root, seq=data)

    print('\nmodel inference')
    feaVecs_query_corse = getFeaVec_DISAM(corse_model, query_imgs,domain=domain_query)
    feaVecs_query_fine = getFeaVec_DISAM(finer_model, query_imgs,domain=domain_query)
    feaVecs_data_corse = getFeaVec_DISAM(corse_model, data_imgs,domain=domain_data)
    feaVecs_data_fine = getFeaVec_DISAM(finer_model, data_imgs,domain=domain_data)


    path = createDir(query, data, domain_query,domain_data)
    f = open(path + 'DISAM_result.txt',mode='w')
    recall_list_c, precision_list_c, recall_list_f, precision_list_f, recall_list_cf, precision_list_cf = \
        recall_precision_n_DISAM_corse_fine(feaVecs_query_corse, feaVecs_data_corse, feaVecs_query_fine, feaVecs_data_fine, recall_num = 25)

    f.write("{}_{}_recall_c:\n".format(query, data))
    f.write(str(recall_list_c) + '\n')
    f.write("{}_{}_precision_c:\n".format(query, data))
    f.write(str(precision_list_c) + '\n')
    f.write("{}_{}_recall_f:\n".format(query, data))
    f.write(str(recall_list_f) + '\n')
    f.write("{}_{}_precision_f:\n".format(query, data))
    f.write(str(precision_list_f) + '\n')
    f.write("{}_{}_recall_cf:\n".format(query, data))
    f.write(str(recall_list_cf) + '\n')
    f.write("{}_{}_precision_cf:\n".format(query, data))
    f.write(str(precision_list_cf) + '\n')

    plotFig(recall_list_cf, precision_list_cf, name=path + "corse_fine/{}_{}".format(query, data))
    plotFig(recall_list_f, precision_list_f, name=path + "fine/{}_{}".format(query, data))
    plotFig(recall_list_c, precision_list_c, name=path + "corse/{}_{}".format(query, data))
    print('finish scene {}_{} domain {} {}'.format(query, data, domain_query, domain_data))

    f.close()

if __name__ == "__main__":

    # root = '../../dataset/SJTU/PR/'
    root = '/media/qzj/Document/grow/research/slamDataSet/SJTU/PR/'
    
    corse_model, finer_model = getDISAM()
    
    for domain_query in range(2,9):
        for domain_data in range(2,9):
            for query in range(1,5):
                for data in range(1,5):
                    test(root, corse_model, finer_model, query, data, domain_query=domain_query, domain_data=domain_data)
