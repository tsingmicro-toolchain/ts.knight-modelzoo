# -*- coding: UTF-8 -*-
import numpy as np
import argparse
import os
import glob
import cv2
import torch
from utils.datasets import LoadStreams, LoadImages

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

from utils.datasets import create_dataloader, LoadImages
from utils.general import (
    coco80_to_coco91_class, check_dataset, check_file, check_img_size, non_max_suppression, 
    xyxy2xywh, xywh2xyxy, box_iou, set_logging)
from tqdm import tqdm

def letterbox_my(im, new_shape=(384, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
class ArgParser:
    def __init__(self, args: dict):
        for k, v in args.items():
            setattr(self, k, v)

if __name__ == '__main__':
    input = FLAGS.input
    outpath = FLAGS.outpath
    proc_mode= FLAGS.proc_mode
    imgsz = FLAGS.img_size
    imgsz = check_img_size(imgsz, s=32)  # check img_size
    # Dataloader
    args_dict = dict(
            data= input,
            single_cls=False,
        )

    opt = ArgParser(args_dict) 
    img = torch.zeros((1, 3, imgsz, imgsz))  # init img
    path = os.path.abspath(input)  # path to val/test images
    #path = os.path.join(os.path.abspath(os.path.join(executor.dataset, '..')),path)
    dataset = LoadImages(input, img_size=640, stride=32)

    input_data = []
    for path, img, im0s, vid_cap in dataset:
        print(img.shape)
        img = torch.from_numpy(img)
        img, ratio, (dw,dh) = letterbox_my(img.permute(1,2,0).numpy(), (640,640), auto=False, stride=32)
        img = torch.from_numpy(img.transpose(2,0,1))
        #img = img.float()  # uint8 to fp16/32
        #img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img = img.numpy()
        img = img.astype(np.uint8)
        input_data.append(img)
    if DEBUG:
        write_infer_numpy_to_file(input_data[img_idx:img_idx+1], os.path.join(outpath,'model_input_all.txt'),True,3)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if proc_mode == 'tf2onnx':
        if DEBUG:
            print(f'SIM IMG:{input_name[img_idx]}')
            write_infer_numpy_to_file(input_data[img_idx:img_idx+1], os.path.join(outpath,'model_input.txt'),True,4)
        data_bin=input_data[img_idx:img_idx+1]
        data_bin.astype(np.float32).flatten().tofile(os.path.join(outpath,"model_input.bin"))
    else:
        input_data = np.array(input_data[0])
        input_data.flatten().tofile(os.path.join(outpath,"model_input.bin"))
    print("Success to save pictures to bin.")