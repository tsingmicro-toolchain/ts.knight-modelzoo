import argparse
import os
from pathlib import Path

import numpy as np
import cv2
import torch
from yolo_nas.models import load_net
from yolo_nas.processing import Preprocessing, Postprocessing, YOLO_NAS_DEFAULT_PROCESSING_STEPS
from yolo_nas.draw import draw_box
from yolo_nas.utils import Labels, export_image, COCO_DEFAULT_LABELS
from onnx_quantize_tool.common.register import onnx_infer_func

iou_thres = 0.45
conf_thres = 0.25
scales = [0.003256865,0.003747179,0.003707352,0.002899949]

def detect(net, source, pre_process, post_process, labels):
    """Detect Image/Frame array"""
    net_input = source.copy()  # copy source array
    input_, prep_meta = pre_process(net_input)  # run preprocess
    input_ = input_.astype(np.uint8)
    # 浮点输入
    #input_, _ = pre_process._standarize(input_, 255)
    
    outputs = net.forward(input_)  # forward
    outputs = [out*scale for out, scale in zip(outputs, scales)]

    # 裁剪部分
    outputs_new = []
    for out in outputs[:-1]:
        out = torch.Tensor(out)
        weight = torch.arange(17).reshape(17,1).float()
        out = out.permute(0,2,3,1).reshape(1, -1, 17)
        out = torch.matmul(out, weight)
        out = out.reshape(1, 1, -1, 4)
        out = out.squeeze(1)
        outputs_new.append(out)
    outputs_new = torch.concat(outputs_new, axis=1)
    split0, split1 = torch.split(outputs_new, 2, dim=-1)

    sub = torch.Tensor(np.load('/ts.knight-modelzoo/pytorch/builtin/cv/detection/yolonas_s/src/_model_heads_Concat_5_output_0_2.npy')) - split0
    add = torch.Tensor(np.load('/ts.knight-modelzoo/pytorch/builtin/cv/detection/yolonas_s/src/_model_heads_Concat_5_output_0.npy')) + split1
    concat = torch.concat((sub, add), axis=-1)
    outputs_new = concat * torch.Tensor(np.load('/ts.knight-modelzoo/pytorch/builtin/cv/detection/yolonas_s/src/_model_heads_Constant_31_output_0.npy'))
    outputs_new = outputs_new.numpy()
    outputs = [outputs_new, outputs[-1]]

    boxes, scores, classes = post_process(outputs, prep_meta)  # postprocess output
    selected = cv2.dnn.NMSBoxes(
        boxes, scores, post_process.conf_thres, post_process.iou_thres
    )  # run nms to filter boxes
    for i in selected:  # loop through selected idx
        box = boxes[i, :].astype(np.int32).flatten()  # get box
        score = float(scores[i]) * 100  # percentage score
        label, color = labels(classes[i], use_bgr=True)  # get label and color class_id
        draw_box(source, box, label, score, color)  # draw boxes
    return source  # Image array after draw process

@onnx_infer_func.register("infer_yolo_nas")
def infer_yolo_nas(executor):
    input_names = executor.input_nodes
    if not executor.shape_dicts:
        executor.init_shape_info()
    input_shapes = executor.shape_dicts
    image = executor.dataset
    save_dir = executor.save_dir
    _, _, input_height, input_width = input_shapes[input_names[0]]
    pre_process = Preprocessing(
        YOLO_NAS_DEFAULT_PROCESSING_STEPS, (input_height, input_width)
    )  # get preprocess
    post_process = Postprocessing(
        YOLO_NAS_DEFAULT_PROCESSING_STEPS,
        iou_thres,
        conf_thres,
    )  # get postprocess

    labels = Labels(COCO_DEFAULT_LABELS)

    img = cv2.imread(image)  # read image
    img = detect(executor, img, pre_process, post_process, labels)  # detect image
    os.makedirs(save_dir,exist_ok=True)
    save_dir = os.path.join(save_dir, os.path.basename(image))
    export_image(img, save_dir)  # export image if configs.export isn't None