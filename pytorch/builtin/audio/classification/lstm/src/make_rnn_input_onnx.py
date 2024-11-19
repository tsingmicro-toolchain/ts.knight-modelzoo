# -*- coding: UTF-8 -*-
import numpy as np
import argparse
import os
import glob
import cv2
import pickle

DEBUG=0
parser = argparse.ArgumentParser(description='make input data scripts')
parser.add_argument(
    '--input',
    type=str,
    default='./',
    help='path of data')

parser.add_argument('--outpath',
                    type=str,
                    default='./',
                    help='the path of output bin')
parser.add_argument('--proc_mode',
                    type=str,
                    default='onnx',
                    help='the proc mode as onnx2onnx')
 
FLAGS, unparsed = parser.parse_known_args()
def load_cmd_data(test_data):
    with open(test_data, 'rb') as fp:
        feat_label =pickle.load(fp,encoding='iso-8859-1')
        test_idx = np.asarray(range(len(feat_label)))
    return feat_label, test_idx

def write_infer_numpy_to_file(data, fileName, is_float=False, dim=4):
    # N=1
    if DEBUG:
        print(f'write_infer_numpy_to_file shape:{data.shape} ,fileName:{fileName}, is_float:{is_float},dim:{dim}')
    N,C, H, W=1,1,1,1
    if dim==4:
        N,C, H, W = data.shape
    else:
        C, H, W = data.shape
    file = open(fileName, 'w+')
    head = 'SHAPE:(N:{0}, C:{1}, H:{2}, W:{3})\n'.format(N, C, H, W)
    file.write(head)
    for i in range(C):
        str_C = ""
        file.write('C[{0}]:\n'.format(i))
        for j in range(H):
            for k in range(W):
                if is_float:
                    if dim==4:
                        str_C = str_C + '{0},'.format(float(data[0][i][j][k]))
                    else:
                        str_C = str_C + '{0},'.format(float(data[i][j][k]))
                else:
                    if dim==4:
                        str_C = str_C + '{0},'.format(int(data[0][i][j][k]))
                    else:
                        str_C = str_C + '{0},'.format(int(data[i][j][k]))
            str_C = str_C + '\n'
        file.write(str_C)
    file.close()

if __name__ == '__main__':
    input = FLAGS.input
    outpath = FLAGS.outpath
    proc_mode= FLAGS.proc_mode
    # input_data = np.load(input).astype(np.float32)
    feat_label,test_idx = load_cmd_data(input)
    data_idx =test_idx
    num_of_batch = int(len(data_idx)/ 2)+ 1
    Num=0
    for i in range(num_of_batch):
        np_feats = np.zeros(shape=(2,25,40),dtype=np.float32)
        ed_label =np.zeros(shape=(2,1),dtype=np.int64)
        for j in range(2):
            idx = data_idx[int(i * 2 + j)% len(data_idx)]
            np_feats[j]= feat_label[idx]['input']
            ed_label[j]= feat_label[idx]['label'].argmax()
        th_feats = np_feats
        ed_label = ed_label

        if DEBUG:
            write_infer_numpy_to_file(th_feats, os.path.join(outpath,'model_input_all.txt'),True,3)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        if proc_mode == 'tf2onnx':
            if DEBUG:
                print(f'SIM IMG:{input_name[img_idx]}')
                write_infer_numpy_to_file(th_feats, os.path.join(outpath,'model_input.txt'),True,4)
            data_bin=th_feats[img_idx:img_idx+1]
            data_bin.astype(np.float32).flatten().tofile(os.path.join(outpath,"model_input.bin"))
        else:
            th_feats.astype(np.float32).flatten().tofile(os.path.join(outpath,"model_input.bin"))
        print("Success to save pictures to bin.")
        if i == Num:
            break