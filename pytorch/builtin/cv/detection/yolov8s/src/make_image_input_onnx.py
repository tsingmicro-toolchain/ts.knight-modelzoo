# -*- coding: UTF-8 -*-
import numpy as np
import argparse
import os
import glob
import cv2
import torch

from ts_utils.general import (
    coco80_to_coco91_class, check_dataset, check_img_size, scale_coords,
    xyxy2xywh, clip_coords, xywh2xyxy, box_iou, set_logging) # non_max_suppression
from ts_utils.metrics import ap_per_class
from onnx_quantize_tool.common.register import onnx_infer_func, pytorch_model
from ts_utils.datasets import LoadImages
from ts_utils.plots import Annotator, colors, save_one_box 

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


if __name__ == '__main__':
    input = FLAGS.input
    outpath = FLAGS.outpath
    proc_mode= FLAGS.proc_mode
    imgsz = FLAGS.img_size
    #imgsz = (640, 640)

    dataset = LoadImages(input, img_size=imgsz)
    for path, im, im0s, vid_cap, s in dataset: 
        # import pdb;pdb.set_trace()
        # if 'zidane' not in path:
        #     continue
        # im = im.float()  # uint8 to fp16/32
        #im = im / 255  # 0 - 255 to 0.0 - 1.0
        #im = im.astype(np.float32)
        if im.ndim == 3:
            im = im[None]  # expand for batch dim  
        im.flatten().tofile(os.path.join(outpath, "model_input.bin"))
