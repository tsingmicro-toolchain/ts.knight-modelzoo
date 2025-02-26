import argparse
import torch
import numpy as np
import os
import cv2
from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import multiclass_nms, demo_postprocess, vis

def detect():
    source, numpys, imgsz, save_dir, scales = opt.image, opt.numpys, opt.img_size, opt.save_dir, opt.scales
    input_shape = (640, 640)
    origin_img = cv2.imread(source)
    img, ratio = preprocess(origin_img, input_shape)
    out0 = np.load(numpys[0])*scales[0]
    out1 = np.load(numpys[1])*scales[1]
    out2 = np.load(numpys[2])*scales[2]
    out0 = torch.from_numpy(out0)
    out1 = torch.from_numpy(out1)
    out2 = torch.from_numpy(out2)
    output = torch.cat([x.flatten(start_dim=2) for x in [out0, out1, out2]], dim=2).permute(0, 2, 1)

    predictions = demo_postprocess(output.numpy(), imgsz)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    scores = predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.15)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=0.9, class_names=COCO_CLASSES)

    output_path = os.path.abspath(os.path.join(save_dir, os.path.basename(source)))
    print(f'save picture to {output_path}')
    cv2.imwrite(output_path, origin_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='original image')
    parser.add_argument('--img-size', type=int, default=(640, 640), help='inference size (pixels)')
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

    detect()