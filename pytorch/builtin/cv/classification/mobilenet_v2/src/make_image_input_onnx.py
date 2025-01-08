# -*- coding: UTF-8 -*-
import numpy as np
import argparse
import os
import glob
import cv2
from torchvision import datasets, transforms
from PIL import Image

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

# 仅将第一条数据进行转换
if __name__ == '__main__':
    input = FLAGS.input
    outpath = FLAGS.outpath
    proc_mode= FLAGS.proc_mode

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    img_rgb = Image.open(input).convert('RGB')
    img_tensor = image_preprocess(img_rgb)
    input_numpy = img_tensor.unsqueeze_(0).numpy()
  
    input_numpy *= 255
    input_numpy = input_numpy.astype(np.uint8)
    print(f'save model_input.bin to {outpath}')
    input_numpy.flatten().tofile(os.path.join(outpath, "model_input.bin"))