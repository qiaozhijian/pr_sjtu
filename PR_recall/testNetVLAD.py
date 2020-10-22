from util.sjtu import sjtu_seq_NetVLAD
from util.evaluate import getNetVLAD,getFeaVec_NetVLAD,recall_precision_n, pool_size, plotFig

if __name__ == "__main__":

    model = getNetVLAD()
    print('load image')
    img_r1 = sjtu_seq_NetVLAD(seq=1)
    img_r2 = sjtu_seq_NetVLAD(seq=2)
    img_r3 = sjtu_seq_NetVLAD(seq=3)
    img_r4 = sjtu_seq_NetVLAD(seq=4)

    print('\nmodel inference')
    feaVecs_r1 = getFeaVec_NetVLAD(model, img_r1, pool_size = pool_size, seq=1)
    feaVecs_r2 = getFeaVec_NetVLAD(model, img_r2, pool_size = pool_size, seq=2)
    feaVecs_r3 = getFeaVec_NetVLAD(model, img_r3, pool_size = pool_size, seq=3)
    feaVecs_r4 = getFeaVec_NetVLAD(model, img_r4, pool_size = pool_size, seq=4)

    feaVecs = [feaVecs_r1, feaVecs_r2, feaVecs_r3, feaVecs_r4]

    for i in range(4):
        for j in range(4):
            if i==j:
                continue
            recall_list, precision_list = recall_precision_n(feaVecs[i], feaVecs[j], recall_num = 25)
            # print('')
            # print(recall_list)
            # print(precision_list)
            plotFig(recall_list, precision_list, name="./figure/NetVLAD/{}_{}".format(i,j))