# -*- coding: UTF-8 -*-
import numpy as np
import argparse
import os
import cv2
from yolox.data.data_augment import preproc as preprocess
from yolox_s import IMAGE_SIZE
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
                    default=(640, 640),
                    help='inference size (pixels)')

FLAGS, unparsed = parser.parse_known_args()

if __name__ == '__main__':
    input = FLAGS.input
    outpath = FLAGS.outpath
    proc_mode= FLAGS.proc_mode
    imgsz = FLAGS.img_size
    origin_img = cv2.imread(input)
    img, ratio = preprocess(origin_img, IMAGE_SIZE)
    img = img.astype(np.uint8)
    img = img[None, :, :, :]
    outpath = os.path.join(outpath, "model_input.bin")

    img.flatten().tofile(outpath)
    print(f"save model_input.bin to {outpath}")