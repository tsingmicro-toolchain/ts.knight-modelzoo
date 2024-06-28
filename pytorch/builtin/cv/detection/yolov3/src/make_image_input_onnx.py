# -*- coding: UTF-8 -*-
import numpy as np
import argparse
import os
import glob
import cv2
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
    # input_data = np.load(input).astype(np.float32)
    input_name = glob.glob(os.path.join(input, "*.jpg"))
    input_name.extend(glob.glob(os.path.join(input, "*.JPEG")))
    input_name.extend(glob.glob(os.path.join(input, "*.png")))
    input_name = sorted(input_name, key=os.path.getctime)
    Num=10

    #match the specified picture for infer demo
    img_idx=4
    assert len(input_name) >= 10, f"expect 10 pictures, but got {len(input_name)}"
    input_data = np.zeros((10, 3, 224, 224), dtype=np.float32)
    if proc_mode == 'caffe2onnx':
        mean = np.array([104.0, 117.0, 123.0]).reshape(1,3,1,1)
        std = np.array([1.0, 1.0, 1.0]).reshape(1, 3, 1, 1)
    else:
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    for i in range(Num):
        img = cv2.cvtColor(cv2.imread(input_name[i]), cv2.COLOR_BGR2RGB)
        r = max(img.shape) / 256
        # for other option
        # interp = cv2.INTER_LINEAR if r < 1 else cv2.INTER_AREA
        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        if proc_mode == 'caffe2onnx':
            img = cv2.resize(img, (256, 256), interpolation=interp)
        else:
            img = cv2.resize(img, (256, 256), interpolation=interp) / 255.
        input_data[i] = img.transpose(2, 0, 1)[:, 15:239, 15: 239]

    input_data = (input_data - mean) / std
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
        input_data.astype(np.float32).flatten().tofile(os.path.join(outpath,"model_input.bin"))
    print("Success to save pictures to bin.")