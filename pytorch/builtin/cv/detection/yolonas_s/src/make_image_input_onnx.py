# -*- coding: UTF-8 -*-
import numpy as np
import argparse
import os
import glob
import cv2
import torch
from yolo_nas.processing import Preprocessing, YOLO_NAS_DEFAULT_PROCESSING_STEPS

DEBUG=0
parser = argparse.ArgumentParser(description='make input data scripts')
parser.add_argument(
    '--input',
    type=str,
    default='./',
    help='path of images')

parser.add_argument('--outpath',
                    type=str,
                    default='./',
                    help='the path of output bin')
parser.add_argument('--proc_mode',
                    type=str,
                    default='onnx',
                    help='the proc mode as onnx2onnx')
parser.add_argument('--img-size',
                    type=int,
                    default=640,
                    help='inference size (pixels)')

FLAGS, unparsed = parser.parse_known_args()

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

class ArgParser:
    def __init__(self, args: dict):
        for k, v in args.items():
            setattr(self, k, v)

if __name__ == '__main__':
    input = FLAGS.input
    outpath = FLAGS.outpath
    proc_mode= FLAGS.proc_mode
    imgsz = FLAGS.img_size
    # Dataloader
    args_dict = dict(
            data= input,
            single_cls=False,
        )

    opt = ArgParser(args_dict) 

    path = os.path.abspath(input)  # path to val/test images
    pre_process = Preprocessing(
        YOLO_NAS_DEFAULT_PROCESSING_STEPS, (imgsz, imgsz)
    )  # get preprocess
    img = cv2.imread(path)
    img, prep_meta = pre_process(img) 

    #img = img.float()  # uint8 to fp16/32
    #img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img.flatten().tofile(os.path.join(outpath,"model_input.bin"))
    print(f"Success to save bin to {outpath}")