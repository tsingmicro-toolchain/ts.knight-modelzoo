from tkinter.messagebox import NO
from matplotlib.pyplot import axes, sca
import torch
import torch.nn as nn
import numpy as np
import cv2
import glob
import os

import json, yaml
from tqdm import tqdm
import time
import torchvision

from pathlib import Path
from ts_utils.general import (
    coco80_to_coco91_class, check_dataset, check_img_size, scale_coords,
    xyxy2xywh, clip_coords, xywh2xyxy, box_iou, set_logging) # non_max_suppression
from ts_utils.metrics import ap_per_class
from onnx_quantize_tool.common.register import onnx_infer_func, pytorch_model
from ts_utils.datasets import LoadImages
from ts_utils.plots import Annotator, colors, save_one_box 
# parameters
config_yolov5s={
'nc': 4,  # number of classes

# anchors
'anchors':[
  [10,13, 16,30, 33,23],  # P3/8
  [30,61, 62,45, 59,119],  # P4/16
  [116,90, 156,198, 373,326]  # P5/32
],

}

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
             59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return x

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes 
    xc = prediction[..., 4] > conf_thres #conf_thres  # candidates # -100

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence 
        # write_numpy_to_file(x.numpy(), "/home/zhengzhe/work/onnx_master/onnx-quantization/onnx_quantize_tool/yoloworld_x_x_x/32_.txt")
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        # x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4]) 
        # Detections matrix nx6 (xyxy, conf, cls)
        if 0:  # x 39 105
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # write_numpy_to_file(boxes.numpy(), "/home/zhengzhe/work/onnx_master/onnx-quantization/onnx_quantize_tool/zhuangbaochun/pre_nms_boxes.txt")
        # write_numpy_to_file(scores.numpy(), "/home/zhengzhe/work/onnx_master/onnx-quantization/onnx_quantize_tool/zhuangbaochun/pre_nms_scores.txt")
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS  # 2881 6
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

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

def dist2bbox(distance, anchor_points, box_format='xyxy', dim=-1):
    '''Transform distance(ltrb) to box(xywh or xyxy).'''
    lt, rb = torch.split(distance, 2, dim) # 1 8400 2, 1 8400 2
    x1y1 = anchor_points - lt # anchor_points 8400 2  x1y1: 1 8400 2
    x2y2 = anchor_points + rb
    if box_format == 'xyxy':
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == 'xywh':
        c_xy = (x1y1 + x2y2) / 2  # [1, 8400, 2]
        wh = x2y2 - x1y1 # [1, 8400, 2]
        bbox = torch.cat([c_xy, wh], dim) # 1 8400 4
    return bbox  # 1, 8400, 4]

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x): 
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors # 1 64 8400
        # return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        constant = torch.arange(self.c1, dtype=torch.float).unsqueeze(1) # 16 1
        x = x.view(b, 4, self.c1, a).softmax(2).permute(0, 1, 3, 2) # 1 64 8400 -> 1 4 16 8400 ->softmax ->permute(1 4 8400 16)
        x = (x @ constant).squeeze(-1) # (1 4 8400 16) matmul (16 1) ->squeeze(1 4 8400)
        return x

def write_numpy_to_file(data, fileName): 
    if data.ndim == 4:
        data = data.squeeze(0) 
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    C,H,W=data.shape
    # import pdb;pdb.set_trace()
    #print(f'nchw:{N},{C},{H},{W}')
    file = open(fileName, 'w+')
    #data_p=data[0][5][2][3]
    #print(f'data:{data_p}')
    #numpy_data = np.array(data)
    # head='SHAPE:(N:{0}, C:{1}, H:{2}, W:{3})\n'.format(N,C,H,W)
    # file.write(head)
    for i in range(C):
        str_C=""
        file.write('C[{0}]:\n'.format(i))
        for j in range(H):
            for k in range(W):
                str_C=str_C+'{:.6f} '.format(data[i][j][k])
                #str_C=str_C+'{0},'.format((data[0][i][j][k]))
            str_C=str_C+'\n'

        file.write(str_C)
    file.close()


''' quant cmd
python run_quantization.py -s /home/zhengzhe/work/onnx_master/onnx-quantization/onnx_quantize_tool/chenfan /
-ch DT53 -if infer_new_yolov8 /
-uds /home/zhengzhe/work/onnx_master/onnx-quantization/onnx_quantize_tool/yolo_repos/ultralytics_main/new_infer_yolov8_easy.py /
-m /home/zhengzhe/work/onnx_master/onnx-quantization/onnx_quantize_tool/chenfan/yolov8s_quantize.onnx -r infer
'''
@onnx_infer_func.register("yolov8s_infer")
def yolov8s_infer(executor):
    
    save_img = True #not nosave and not source.endswith('.txt')  # save inference images

    # Directories 
    save_dir = executor.save_dir

    if executor.run_mode == "forward":
        img_path = "/tmp/test_data"
    img_path = executor.dataset
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    # Load model
    run = executor.forward
    imgsz = (640, 640)
   


    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
        


    nc = 80
    concat_dim = 145
    # imgsz = check_img_size(imgsz, s=stride)  # check image size
    save_crop = False
    save_conf = True
    hide_labels = False
    hide_conf = False
    line_thickness = 1
    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(img_path, img_size=imgsz)
    vid_path, vid_writer = [None] * bs, [None] * bs
    dfl = DFL(16)
    # Run inference
    count = 0
    save_txt = False
    for path, im, im0s, vid_cap, s in dataset:
        # import pdb;pdb.set_trace()
        # if 'zidane' not in path:
        #     continue
        # im = im.float()  # uint8 to fp16/32
        #im = im / 255  # 0 - 255 to 0.0 - 1.0
        #im = im.astype(np.float32)
        if im.ndim == 3:
            im = im[None]  # expand for batch dim  
        im.flatten().tofile(os.path.join(executor.save_dir, "model_input.bin"))
        cls0, cls1, cls2 = run(im) #, npys[count].astype(np.float32))   # [1 165 80 80] [1 165 40 40] [1 165 20 20]  
        write_numpy_to_file(cls0, os.path.join(executor.save_dir, "699.txt"))
        write_numpy_to_file(cls1, os.path.join(executor.save_dir, "719.txt"))
        write_numpy_to_file(cls2, os.path.join(executor.save_dir, "739.txt"))
        # continue
        # names = _names[count]

        count += 1
        # raise
        cls0 = torch.from_numpy(cls0)
        cls1 = torch.from_numpy(cls1)
        cls2 = torch.from_numpy(cls2)  
        cls0[:,64:,...] = cls0[:,64:,...].sigmoid() # 前64是框的信息 
        cls1[:,64:,...] = cls1[:,64:,...].sigmoid()
        cls2[:,64:,...] = cls2[:,64:,...].sigmoid() 
        outputs = torch.cat([cls_.reshape(1, concat_dim, -1) for cls_ in [cls0, cls1, cls2]], dim=-1) # 1 165  8400   
        anchor_points = torch.from_numpy(np.load("/ts.knight-modelzoo/pytorch/builtin/cv/detection/yolov8s/src/anchor_points.npy"))
        anchor_points = anchor_points.permute(1,0) 
        stride_tensor = torch.from_numpy(np.load("/ts.knight-modelzoo/pytorch/builtin/cv/detection/yolov8s/src/stride_tensor.npy"))
        stride_tensor = stride_tensor.permute(1, 0)  
        box, cls = outputs.split((16 * 4, nc + 1), 1)
        # 64 * 6400 -> 4 * 6400
        dbox = dist2bbox(dfl(box), anchor_points.unsqueeze(0), 'xywh',dim=1) * stride_tensor
        #print(path) # 165 80 80 
        outputs = torch.cat((
            dbox, 
            cls
            ), 
            1).permute(0, 2 ,1)
        #print(path) 
        pred = non_max_suppression(outputs, 0.2, 0.7) # 0.2 
        for i, det in enumerate(pred):  # per image 300*6
            p, im0 = path, im0s.copy() 
            p = Path(p)  # to Path
            # import pdb;pdb.set_trace()
            if save_dir:
                save_path = os.path.join(save_dir, p.name)  # im.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example="test")
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, 5].unique(): # 100个类 300 x 6
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    #print(s)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if 0:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop:  # Add bbox to image
                        c = int(cls)  # integer class 
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        print("x1,y1,x2,y2", xyxy)

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    if save_dir:
                        cv2.imwrite(save_path, im0)
                        print(f'\nsave picture to {save_path}')
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)  
        break
    return None
