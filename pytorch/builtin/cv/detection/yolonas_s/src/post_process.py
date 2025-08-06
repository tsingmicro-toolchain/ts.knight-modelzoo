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

def detect(net, source, pre_process, post_process, labels):
    """Detect Image/Frame array"""
    net_input = source.copy()  # copy source array
    input_, prep_meta = pre_process(net_input)  # run preprocess
    
    # 浮点输入
    input_, _ = pre_process._standarize(input_, 255)
    
    if net is not None:
        outputs = net.forward(input_)  # forward
        outputs = [out*scale for out, scale in zip(outputs, opt.scales)]
    else:
        outputs = [np.load(out)*scale for out, scale in zip(opt.numpys, opt.scales)]

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

    sub = torch.Tensor(np.load(os.path.join(os.path.dirname(__file__),'_model_heads_Concat_5_output_0_2.npy'))) - split0
    add = torch.Tensor(np.load(os.path.join(os.path.dirname(__file__),'_model_heads_Concat_5_output_0.npy'))) + split1
    concat = torch.concat((sub, add), axis=-1)
    outputs_new = concat * torch.Tensor(np.load(os.path.join(os.path.dirname(__file__),'_model_heads_Constant_31_output_0.npy')))
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


def main():
    if opt.model:
        net = load_net(opt.model)  # load net
        net.warmup()  # warmup net
        _, _, input_height, input_width = net.input_shape  # get input height and width [b, c, h, w]
    else:
        net = None
        input_height = input_width = opt.img_size
    pre_process = Preprocessing(
        YOLO_NAS_DEFAULT_PROCESSING_STEPS, (input_height, input_width)
    )  # get preprocess
    post_process = Postprocessing(
        YOLO_NAS_DEFAULT_PROCESSING_STEPS,
        opt.iou_thres,
        opt.conf_thres,
    )  # get postprocess

    labels = Labels(COCO_DEFAULT_LABELS)

    img = cv2.imread(opt.image)  # read image
    img = detect(net, img, pre_process, post_process, labels)  # detect image
    os.makedirs(opt.save_dir,exist_ok=True)
    save_dir = os.path.join(opt.save_dir, os.path.basename(opt.image))
    export_image(img, save_dir)  # export image if configs.export isn't None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect using YOLO-NAS model")
    parser.add_argument("-m", "--model", type=str, help="YOLO-NAS ONNX model path")
    parser.add_argument('--image', type=str, help='original image')
    parser.add_argument("-v", "--video", type=str, help="Video source")
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--numpys', nargs='+', type=str, help='model output numpy')
    parser.add_argument('--scales', nargs='+', type=float, help='model output scales')
    parser.add_argument('--save_dir', type=str, default='output', help='save dir for detect')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument(
        "--export",
        type=str,
        help="Export to a file (path with extension | mp4 is a must for video)",
    )
    opt = parser.parse_args()

    main()
