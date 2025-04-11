# -*- coding: UTF-8 -*-
import numpy as np
import argparse
import os
import glob
import cv2
import torch
from tqdm import tqdm
from yolov6.data.datasets import LoadData
from yolov6s import IMAGE_SIZE
def letterbox_my(im, new_shape=(384, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
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

def process_image(img_src, img_size, stride, half):
    '''Process image before image inference.'''
    image = letterbox_my(img_src, img_size, stride=stride)[0]
    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image, img_src 

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
    files = LoadData(input, False, 0)
    for img_src, img_path, vid_cap in tqdm(files):
        img, img_src = process_image(img_src, IMAGE_SIZE, 32, False)
        if len(img.shape) == 3:
            img = img[None]
            # expand for batch dim
        img *= 255
        img = img.numpy().astype(np.uint8)
        img.flatten().tofile(os.path.join(outpath, "model_input.bin"))
        print(f"save make_input.bin to {outpath}")
