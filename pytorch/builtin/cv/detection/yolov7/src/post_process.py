import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import os
import re
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

# parameters
config_yolov7_tiny_relu={
'nc': 80,  # number of classes
'depth_multiple': 0.33,  # model depth multiple
'width_multiple': 0.50,  # layer channel multiple

# anchors
'anchors':[
  [10,13, 16,30, 33,23],  # P3/8
  [30,61, 62,45, 59,119],  # P4/16
  [116,90, 156,198, 373,326]  # P5/32
],

# YOLOv backbone
'backbone':
  # [from, number, module, args]
  [[-1, 1, 'Conv', [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
   [-1, 3, 'C3', [128]],
   [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
   [-1, 6, 'C3', [256]],
   [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
   [-1, 9, 'C3', [512]],
   [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
   [-1, 3, 'C3', [1024]],
   [-1, 1, 'SPPF', [1024, 5]],  # 9
  ],

# YOLOv5 head
'head':
 [[-1, 1, 'Conv', [512, 1, 1]],
   [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
   [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
   [-1, 3, 'C3', [512, False]],  # 13

   [-1, 1, 'Conv', [256, 1, 1]],
   [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
   [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
   [-1, 3, 'C3', [256, False]],  # 17 (P3/8-small)

   [-1, 1, 'Conv', [256, 3, 2]],
   [[-1, 14], 1, 'Concat', [1]],  # cat head P4
   [-1, 3, 'C3', [512, False]],  # 20 (P4/16-medium)

   [-1, 1, 'Conv', [512, 3, 2]],
   [[-1, 10], 1, 'Concat', [1]],  # cat head P5
   [-1, 3, 'C3', [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, 'Detect', ['nc', 'anchors']],  # Detect(P3, P4, P5)
  ],
}
class Detect_process(nn.Module):
    def __init__(self, anchors, nc=1, nl=3, ch=3, na=3, inplace=True):
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2)  # shape(nl,na,2)

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.stride = torch.Tensor([8, 16, 32])
        self.anchors /= self.stride.view(-1, 1, 1)


    def post_process(self, x):
        z = []  # inference output
        for i in range(len(x)):
            # x[i] = torch.nn.functional.sigmoid(x[i])
            bs, _, ny, nx, _ = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            #x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            y = x[i].sigmoid()
            if self.inplace:
                y[..., 0:2] = (y[..., 0:2] * 2. + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))
        return torch.cat(z, 1)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i]
        shape = 1, self.na, ny, nx, 2  # grid shape
        yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx))
        grid = torch.stack((xv, yv), 2).expand(shape).float() - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape).float()
        return grid, anchor_grid

def detect(save_img=False):
    source, numpys, imgsz, save_dir, scales = opt.image, opt.numpys, opt.img_size, opt.save_dir, opt.scales
    dataset = LoadImages(source, img_size=imgsz, stride=32)
    detect_process = Detect_process(config_yolov7_tiny_relu['anchors'], nc=config_yolov7_tiny_relu['nc'])
    # Get names and colors
    names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
    random.seed(1)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = []
        for numpy, scale in zip(numpys, scales):
            pred.append(torch.from_numpy(np.array([np.load(numpy)])*scale))
        pred = detect_process.post_process(pred)
        #pred = torch.from_numpy(pred[0])
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = os.path.join(save_dir, p.name)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords((imgsz, imgsz), det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)


            cv2.imwrite(save_path, im0)
            print(f" The image with the result is saved in: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='original image')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--numpys', nargs='+', type=str, help='model output numpy')
    parser.add_argument('--scales', nargs='+', type=float, help='model output scales')
    parser.add_argument('--save_dir', type=str, default='output', help='save dir for detect')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    detect()