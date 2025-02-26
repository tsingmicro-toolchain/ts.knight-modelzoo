import torch
import torch.nn as nn
import numpy as np
import cv2
import glob
import os
import shutil
from onnx_quantize_tool.common.register import onnx_infer_func, pytorch_model
import json, yaml
from tqdm import tqdm
from numpy import random
import time
import argparse

from pathlib import Path
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, 
                           coco80_to_coco91_class, non_max_suppression, 
                           scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.torch_utils import select_device, time_sync


# parameters
config_yolov3_tiny={
'nc': 80,  # number of classes
'depth_multiple': 1.0,  # model depth multiple
'width_multiple': 1.0,  # layer channel multiple

# anchors
'anchors':[
  [10,14, 23,27, 37,58],  # P4/16
  [81,82, 135,169, 344,319],  # P5/32
],

# YOLOv3-tiny backbone
'backbone':
  # [from, number, module, args]
  [[-1, 1, 'Conv', [16, 3, 1]],  # 0
   [-1, 1, 'nn.MaxPool2d', [2, 2, 0]],  # 1-P1/2
   [-1, 1, 'Conv', [32, 3, 1]],
   [-1, 1, 'nn.MaxPool2d', [2, 2, 0]],  # 3-P2/4
   [-1, 1, 'Conv', [64, 3, 1]],
   [-1, 1, 'nn.MaxPool2d', [2, 2, 0]],  # 5-P3/8
   [-1, 1, 'Conv', [128, 3, 1]],
   [-1, 1, 'nn.MaxPool2d', [2, 2, 0]],  # 7-P4/16
   [-1, 1, 'Conv', [256, 3, 1]],
   [-1, 1, 'nn.MaxPool2d', [2, 2, 0]],  # 9-P5/32
   [-1, 1, 'Conv', [512, 3, 1]],
   [-1, 1, 'nn.ZeroPad2d', [[0, 1, 0, 1]]],  # 11
   [-1, 1, 'nn.MaxPool2d', [2, 1, 0]],  # 12
  ],

# YOLOv3-tiny head
'head':
  [[-1, 1, 'Conv', [1024, 3, 1]],
   [-1, 1, 'Conv', [256, 1, 1]],
   [-1, 1, 'Conv', [512, 3, 1]],  # 15 (P5/32-large)

   [-2, 1, 'Conv', [128, 1, 1]],
   [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
   [[-1, 8], 1, 'Concat', [1]],  # cat backbone P4
   [-1, 1, 'Conv', [256, 3, 1]],  # 19 (P4/16-medium)

   [[19, 15], 1, 'Detect', ['nc', 'anchors']],  # Detect(P4, P5)
  ]
}

# parameters
config_yolov3={
'nc': 80,  # number of classes
'depth_multiple': 1.0,  # model depth multiple
'width_multiple': 1.0,  # layer channel multiple

# anchors
'anchors':[
  [10,13, 16,30, 33,23],  # P3/8
  [30,61, 62,45, 59,119],  # P4/16
  [116,90, 156,198, 373,326],  # P5/32
],

# darknet53 backbone
'backbone':
  # [from, number, module, args]
  [[-1, 1, 'Conv', [32, 3, 1]],  # 0
   [-1, 1, 'Conv', [64, 3, 2]],  # 1-P1/2
   [-1, 1, 'Bottleneck', [64]],
   [-1, 1, 'Conv', [128, 3, 2]],  # 3-P2/4
   [-1, 2, 'Bottleneck', [128]],
   [-1, 1, 'Conv', [256, 3, 2]],  # 5-P3/8
   [-1, 8, 'Bottleneck', [256]],
   [-1, 1, 'Conv', [512, 3, 2]],  # 7-P4/16
   [-1, 8, 'Bottleneck', [512]],
   [-1, 1, 'Conv', [1024, 3, 2]],  # 9-P5/32
   [-1, 4, 'Bottleneck', [1024]],  # 10
  ],

# YOLOv3 head
'head':
   [[-1, 1, 'Bottleneck', [1024, False]],
   [-1, 1, 'Conv', [512, [1, 1]]],
   [-1, 1, 'Conv', [1024, 3, 1]],
   [-1, 1, 'Conv', [512, 1, 1]],
   [-1, 1, 'Conv', [1024, 3, 1]],  # 15 (P5/32-large)

   [-2, 1, 'Conv', [256, 1, 1]],
   [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
   [[-1, 8], 1, 'Concat', [1]],  # cat backbone P4
   [-1, 1, 'Bottleneck', [512, False]],
   [-1, 1, 'Bottleneck', [512, False]],
   [-1, 1, 'Conv', [256, 1, 1]],
   [-1, 1, 'Conv', [512, 3, 1]],  # 22 (P4/16-medium)

   [-2, 1, 'Conv', [128, 1, 1]],
   [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
   [[-1, 6], 1, 'Concat', [1]],  # cat backbone P3
   [-1, 1, 'Bottleneck', [256, False]],
   [-1, 2, 'Bottleneck', [256, False]],  # 27 (P3/8-small)

   [[27, 22, 15], 1, 'Detect', ['nc', 'anchors']],   # Detect(P3, P4, P5)
  ]
}

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

def affine_box(dets, ratio, dw, dh):
    for box in dets:
        box[0]= int((box[0]-dw)/ratio[0])
        box[1]= int((box[1]-dh)/ratio[1])
        box[2]= int((box[2]-dw)/ratio[0])
        box[3]= int((box[3]-dh)/ratio[1])
    return dets
def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        for i in range(self.nl):
            x[i]= self.m[i](x[i])
            bs,_,ny,nx =x[i].shape
            x[i]=( x[i]
                  .view(bs, self.na, self.no, ny, nx)
                  .permute(0,1,3,4,2)
                  .contiguous()
            )
        return x

@pytorch_model.register("yolov3")
def yolov3(weight_path=None):
    from models.common import DetectMultiBackend
    if weight_path:
        model = DetectMultiBackend(weight_path, device=torch.device('cpu'), dnn=False)
    model.eval()
    in_dict = {
        "model": model,
        "inputs": [torch.randn(1, 3, 640, 640)],
    }
    return in_dict

@onnx_infer_func.register("yolov3_quant")
def yolov3_quant(executor):
    if executor.run_mode == "forward":
        data='coco.yaml'
    else:
        data='coco128.yaml'
    args_dict = dict(
        data= data,
        single_cls=False,
    )   
    iteration = executor.iteration
    save_dir = executor.dataset
    device='cpu'
    task='val'
    imgsz = 640
    training = False
    device = 'cpu'
    single_cls = False
    conf_thres=0.001
    iou_thres=0.6
    max_det=300
    verbose=False
    save_hybrid = False
    plots = False
    callbacks=Callbacks()
    opt = ArgParser(args_dict) 
    data = executor.dataset
    if data=='coco.yaml':
        save_json=True
    else:
        save_json=False

    # Load model
    device = select_device(device, batch_size=1)
    imgsz = check_img_size(imgsz, s=32)  # check img_size
    names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    #check_dataset(data)  # check
    nc = 1 if opt.single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    # Dataloader
    if not training:
        im = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init im
        pad = 0.0
        path = data['val']  # path to val/test images
        path = os.path.join(os.path.abspath(os.path.join(executor.dataset, '..')),path)
        data['val'] = path
        dataloader = create_dataloader(data[task], imgsz, 1, 32, single_cls, pad=pad, rect=True)[0]
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    class_map  = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        im0 = im
        nb, _, height, width = im0.shape  # batch size, channels, height, width
        im, ratio, (dw,dh) = letterbox_my(im.squeeze(0).permute(1,2,0).numpy(), (640,640), auto=False, stride=32)
        im = torch.from_numpy(im.transpose(2,0,1))
        #im = im.float()  # uint8 to fp16/32
        #im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
        # Inference
        t1 = time_sync()
        preds = executor.forward(im.numpy())
        nms = True
        if nms:
            root = os.path.dirname(os.path.abspath(__file__))
            tensor0_0 = torch.from_numpy(np.load(os.path.join(root, 'tensor0_0.npy')))
            tensor0_1 = torch.from_numpy(np.load(os.path.join(root, 'tensor0_1.npy')))
            tensor1_0 = torch.from_numpy(np.load(os.path.join(root, 'tensor1_0.npy')))
            tensor1_1 = torch.from_numpy(np.load(os.path.join(root, 'tensor1_1.npy')))
            out = torch.from_numpy(preds[0])
            out = torch.sigmoid(out)
            out0, out1, out2 = torch.split(out, [2,2,81], dim=-1)
            out0 = (out0*2+tensor0_0)*16
            out1 = (out1*2)**2*tensor0_1
            outs0 = torch.cat([out0, out1, out2], dim=-1)
            outs0 = outs0.reshape((1, 4800, 85))

            out = torch.from_numpy(preds[1])
            out = torch.sigmoid(out)
            out0, out1, out2 = torch.split(out, [2,2,81], dim=-1)
            out0 = (out0*2+tensor1_0)*32
            out1 = (out1*2)**2*tensor1_1
            outs1 = torch.cat([out0, out1, out2], dim=-1)
            outs1 = outs1.reshape((1,1200,85))
            output = torch.cat([outs0, outs1], dim=1)
        preds = [output, torch.from_numpy(preds[0]), torch.from_numpy(preds[1])] 
        #pred = [torch.from_numpy(preds[0]), torch.from_numpy(pred[1]s)]
        # NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height])  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling

        preds = non_max_suppression(
            preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
        )
        t2 = time_sync()
        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)

            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])


        callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds)
        if batch_i >= iteration:
            break

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy

    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING no labels found in {task} set, can not compute metrics without labels")

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return
