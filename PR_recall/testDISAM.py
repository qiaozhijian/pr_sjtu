from util.sjtu import sjtu_seq_DISAM
from util.evaluate import getDISAM,getFeaVec_DISAM,recall_precision_n_DISAM, plotFig,recall_precision_n_DISAM_corse_fine
import torch
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    corse_model, finer_model = getDISAM()
    print('load image')
    img_r1 = sjtu_seq_DISAM(seq=1)
    img_r2 = sjtu_seq_DISAM(seq=2)
    img_r3 = sjtu_seq_DISAM(seq=3)
    img_r4 = sjtu_seq_DISAM(seq=4)

    print('\nmodel inference')
    feaVecs_r1_corse = getFeaVec_DISAM(corse_model, img_r1, seq=1, mode='corse')
    feaVecs_r2_corse = getFeaVec_DISAM(corse_model, img_r2, seq=2, mode='corse')
    feaVecs_r3_corse = getFeaVec_DISAM(corse_model, img_r3, seq=3, mode='corse')
    feaVecs_r4_corse = getFeaVec_DISAM(corse_model, img_r4, seq=4, mode='corse')

    feaVecs_r1_finer = getFeaVec_DISAM(finer_model, img_r1, seq=1, mode='finer')
    feaVecs_r2_finer = getFeaVec_DISAM(finer_model, img_r2, seq=2, mode='finer')
    feaVecs_r3_finer = getFeaVec_DISAM(finer_model, img_r3, seq=3, mode='finer')
    feaVecs_r4_finer = getFeaVec_DISAM(finer_model, img_r4, seq=4, mode='finer')

    feaVecs_corse = [feaVecs_r1_corse, feaVecs_r2_corse, feaVecs_r3_corse, feaVecs_r4_corse]
    feaVecs_finer = [feaVecs_r1_finer, feaVecs_r2_finer, feaVecs_r3_finer, feaVecs_r4_finer]

    f = open('DISAM_result.txt',mode='w')
    for i in range(4):
        for j in range(4):
            if i==j: feaVecs_finer
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

            plotFig(recall_list_cf, precision_list_cf, name="./figure/DISAM/corse_fine/{}_{}".format(i,j))
            plotFig(recall_list_f, precision_list_f, name="./figure/DISAM/fine/{}_{}".format(i,j))
            plotFig(recall_list_c, precision_list_c, name="./figure/DISAM/corse/{}_{}".format(i,j))
            print('finish {}_{}'.format(i,j))
    f.close()