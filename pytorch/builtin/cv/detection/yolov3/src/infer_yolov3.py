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
from utils.datasets import create_dataloader
from utils.general import (NCOLS, box_iou, check_dataset, check_img_size, 
                           coco80_to_coco91_class, non_max_suppression, 
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ap_per_class
from utils.torch_utils import select_device, time_sync
from utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)


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
        # yolov3_tiny
        self.stride = torch.Tensor([16, 32])
        # # yolov3
        # self.stride = torch.Tensor([8, 16, 32])
        self.anchors /= self.stride.view(-1, 1, 1)



    def post_process(self, x):
        z = []  # inference output
        for i in range(len(x)):
            # x[i] = torch.nn.functional.sigmoid(x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

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
    concre_args = {'augment': False, 'profile': False, "val": True}
    name = list(model.model.model.named_children())[-1]
    identity = Identity()
    detect = getattr(model.model.model,name)
    identity.__dict__.update(detect.__dict__)
    setattr(model.model.model,name, identity)
    in_dict = {
        "model": model.model,
        "inputs": [torch.randn(1, 3, 416, 416)],
        "concre_args": concre_args
    }
    return in_dict

@onnx_infer_func.register("infer_yolov3")
def infer_yolov3(executor):
    if executor.run_mode == "forward":
        data='coco.yaml'
    else:
        data='coco128.yaml'
    args_dict = dict(
        data= data,
        single_cls=False,
    )   
    iteration = executor.iteration
    device='cpu'
    task='val'
    imgsz = 416
    training = False
    device = 'cpu'
    single_cls = False
    opt = ArgParser(args_dict) 
    data = opt.data
    if data=='coco.yaml':
        save_json=True
    else:
        save_json=False
    

    # Load model
    device = select_device(device, batch_size=1)
    imgsz = check_img_size(imgsz, s=32)  # check img_size
    detect_process = Detect_process(config_yolov3_tiny['anchors'], nc=config_yolov3_tiny['nc'])
    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if opt.single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        pad = 0.0
        path = data['val']  # path to val/test images
        dataloader = create_dataloader(data[task], imgsz, 1, 32, single_cls, pad=pad, rect=True)[0]
    seen = 0
    class_map  = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (img, targets, paths, shapes) in enumerate(pbar):
        im0 = img
        nb, _, height, width = im0.shape  # batch size, channels, height, width
        img, ratio, (dw,dh) = letterbox_my(img.squeeze(0).permute(1,2,0).numpy(), (416,416), auto=False, stride=32)
        img = torch.from_numpy(img.transpose(2,0,1))
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_sync()
        pred = executor.forward(img.numpy())
        pred = [torch.from_numpy(pred[0]), torch.from_numpy(pred[1])]
        pred = detect_process.post_process(pred)
        # Apply NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height])  # to pixels
        output = non_max_suppression(pred, 0.001, 0.6)
        t2 = time_sync()
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            pred = affine_box(pred, ratio, dw, dh)
            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
            # Append to pycocotools JSON dictionary
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            

        if batch_i > iteration:
            break

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Save JSON
    if save_json and len(jdict):
        f = 'detections_val2017_yolov3_tiny_relu_results.json'  # filename
        print('\nCOCO mAP with pycocotools... saving %s...' % f)
        with open(f, 'w') as file:
            json.dump(jdict, file)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            cocoGt = COCO(glob.glob('/data/public_data/data_coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes(f)  # initialize COCO pred api
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # image IDs to evaluate
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # return (mp, mr, map50, map), maps, t
    return map50
