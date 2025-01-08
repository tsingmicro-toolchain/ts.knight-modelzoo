import torch
import torch.nn as nn
import numpy as np
import cv2
import glob
import os
import shutil
import time
import argparse
import json, yaml
from tqdm import tqdm
from numpy import random
from onnx_quantize_tool.common.register import onnx_infer_func, pytorch_model
from pathlib import Path
from utils.dataloaders import create_dataloader, LoadImages
from utils.general import (
    coco80_to_coco91_class, check_file, check_img_size, non_max_suppression, 
    xyxy2xywh, xywh2xyxy, box_iou, set_logging, scale_boxes)
from utils.metrics import ap_per_class 
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.plots import Annotator, colors, save_one_box

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
    
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
# parameters
config_yolov5s_relu={
'nc': 80,  # number of classes
'depth_multiple': 0.33,  # model depth multiple
'width_multiple': 0.50,  # layer channel multiple

# anchors
'anchors':[
  [10,13, 16,30, 33,23],  # P3/8
  [30,61, 62,45, 59,119],  # P4/16
  [116,90, 156,198, 373,326]  # P5/32
],

# YOLOv5 backbone
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
        self.stride = torch.Tensor([8, 16, 32])
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

class Identify(torch.nn .Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
        return x

@pytorch_model.register("yolov5s_7")
def yolov5s_7(weight_path=None):
    if weight_path:
        model = DetectMultiBackend(weight_path, device='cpu', dnn=False)
    name,_= list(model.model.model.named_children())[-1]
    identity = Identify()
    detect = getattr(model.model.model, name)
    identity.__dict__.update(detect.__dict__)
    setattr(model.model.model, name, identity)
    in_dict = {
        "model": model.model,
        "inputs": [torch.randn((1,3,640,640))],
        "concrete_args":{"augment":False,"profile":False,"visualize":False,"val":True}
    }
    return in_dict

@onnx_infer_func.register("yolov5s_quant")
def yolov5s_quant(executor): 
    data = executor.dataset # /path/to/your/coco128.yaml
    save_json=False
    if executor.run_mode == "forward":
        save_json=True
    args_dict = dict(
        data= data,
        single_cls=False,
    )
    iteration = executor.iteration
    batch_size = executor.batch_size
    img_size=640
    device='cpu'
    task='val'
    augment=False
    merge=False
    verbose=False
    save_txt=False
    imgsz = 640
    training = False
    device = 'cpu'
    opt = ArgParser(args_dict) 
    data = opt.data 
    set_logging()
    
    # Load model
    set_logging()
    device = select_device(device, batch_size=1)
    imgsz = check_img_size(imgsz, s=32)  # check img_size
    detect_process = Detect_process(config_yolov5s_relu['anchors'], nc=config_yolov5s_relu['nc'])

    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict 
    nc = 1 if opt.single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        path = data['val']  # path to val/test images
        path = os.path.join(os.path.abspath(os.path.join(executor.dataset, '..')),path)
        dataloader = create_dataloader(path, imgsz, 1, 32, opt,
                                       hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]
    seen = 0
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, = 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        im0 = img
        nb, _, height, width = im0.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height])
        img, ratio, (dw,dh) = letterbox_my(img.squeeze(0).permute(1,2,0).numpy(), (640,640), auto=False, stride=32)
        img = torch.from_numpy(img.transpose(2,0,1))
        #img = img.float()  # uint8 to fp16/32
        #img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = executor.forward(img.numpy().astype(np.uint8))
        pred = [torch.from_numpy(pred[0]), torch.from_numpy(pred[1]), torch.from_numpy(pred[2])]
        pred = detect_process.post_process(pred)
        # Apply NMS
        output = non_max_suppression(pred, 0.001, 0.65)
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Clip boxes to image bounds
            pred = affine_box(pred, ratio, dw, dh)
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = Path(paths[si]).stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': int(image_id) if image_id.isnumeric() else image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
        if batch_i > iteration:
            break

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, fl, ap, ap_class = ap_per_class(*stats, names={})
        ap50, ap = ap[:,0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Save JSON
    if save_json and len(jdict):
        f = 'detections_val2017_yolov5s_v7_0_relu_results.json'  # filename
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

@onnx_infer_func.register("yolov5s_infer")
def yolov5s_infer(executor): 
    data = executor.dataset # /path/to/your/coco128.yaml
    iteration = executor.iteration
    batch_size = executor.batch_size
    device='cpu'
    task='val'
    augment=False
    merge=False
    verbose=False
    save_txt=False
    imgsz = 640
    training = False

    input_data = []
    dataset = LoadImages(data, img_size=imgsz, stride=32)
    detect_process = Detect_process(config_yolov5s_relu['anchors'], nc=config_yolov5s_relu['nc'])
    for path, img, im0s, vid_cap, s in dataset:
        print(img.shape)
        img = torch.from_numpy(img)
        img, ratio, (dw,dh) = letterbox_my(img.permute(1,2,0).numpy(), (640,640), auto=False, stride=32)
        img = torch.from_numpy(img.transpose(2,0,1))
        #img = img.float()  # uint8 to fp16/32
        #img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
 
        # Inference
        pred = executor.forward(img.numpy())
        pred = [torch.from_numpy(pred[0]), torch.from_numpy(pred[1]), torch.from_numpy(pred[2])]
        pred = detect_process.post_process(pred)
        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = os.path.join(executor.save_dir, p.name)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = (f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
            im0 = annotator.result()

            cv2.imwrite(save_path, im0)
            print(f" The image with the result is saved in: {save_path}")


