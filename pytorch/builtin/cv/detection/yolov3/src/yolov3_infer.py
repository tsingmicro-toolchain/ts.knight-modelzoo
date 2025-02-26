# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""
Run YOLOv3 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
import numpy as np
from onnx_quantize_tool.common.register import onnx_infer_func

@onnx_infer_func.register("yolov3_infer")
def yolov3_infer(executor):
    weights=ROOT / "yolov5s.pt"  # model path or triton URL
    source= executor.dataset
    save_dir = executor.save_dir
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640)  # inference size (height, width)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    device=""  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False  # show results
    save_txt=False  # save results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    update=False  # update all models
    project=ROOT / "runs/detect"  # save results to project/name
    name="exp"  # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    dnn=False  # use OpenCV DNN for ONNX inference
    vid_stride=1  # video frame-rate stride

    """
    Run YOLOv3 detection inference on various input sources such as images, videos, streams, and YouTube URLs.

    Args:
        weights (str | Path): Path to the model weights file or a Triton URL (default: 'yolov5s.pt').
        source (str | Path): Source of input data such as a file, directory, URL, glob pattern, or device identifier
            (default: 'data/images').
        data (str | Path): Path to the dataset YAML file (default: 'data/coco128.yaml').
        imgsz (tuple[int, int]): Inference size as a tuple (height, width) (default: (640, 640)).
        conf_thres (float): Confidence threshold for detection (default: 0.25).
        iou_thres (float): Intersection Over Union (IOU) threshold for Non-Max Suppression (NMS) (default: 0.45).
        max_det (int): Maximum number of detections per image (default: 1000).
        device (str): CUDA device identifier, e.g., '0', '0,1,2,3', or 'cpu' (default: '').
        view_img (bool): Whether to display results during inference (default: False).
        save_txt (bool): Whether to save detection results to text files (default: False).
        save_conf (bool): Whether to save detection confidences in the text labels (default: False).
        save_crop (bool): Whether to save cropped detection boxes (default: False).
        nosave (bool): Whether to prevent saving images or videos with detections (default: False).
        classes (list[int] | None): List of class indices to filter, e.g., [0, 2, 3] (default: None).
        agnostic_nms (bool): Whether to perform class-agnostic NMS (default: False).
        augment (bool): Whether to apply augmented inference (default: False).
        visualize (bool): Whether to visualize feature maps (default: False).
        update (bool): Whether to update all models (default: False).
        project (str | Path): Path to the project directory where results will be saved (default: 'runs/detect').
        name (str): Name for the specific run within the project directory (default: 'exp').
        exist_ok (bool): Whether to allow existing project/name directory without incrementing run index (default: False).
        line_thickness (int): Thickness of bounding box lines in pixels (default: 3).
        hide_labels (bool): Whether to hide labels in the results (default: False).
        hide_conf (bool): Whether to hide confidences in the results (default: False).
        half (bool): Whether to use half-precision (FP16) for inference (default: False).
        dnn (bool): Whether to use OpenCV DNN for ONNX inference (default: False).
        vid_stride (int): Stride for video frame rate (default: 1).

    Returns:
        None

    Notes:
        This function supports a variety of input sources such as image files, video files, directories, URL patterns,
        webcam streams, and YouTube links. It also supports multiple model formats including PyTorch, ONNX, OpenVINO,
        TensorRT, CoreML, TensorFlow, PaddlePaddle, and others. The results can be visualized in real-time or saved to
        specified directories. Use command-line arguments to modify the behavior of the function.

    Examples:
        ```python
        # Run YOLOv3 inference on an image
        run(weights='yolov5s.pt', source='data/images/bus.jpg')

        # Run YOLOv3 inference on a video
        run(weights='yolov5s.pt', source='data/videos/video.mp4', view_img=True)

        # Run YOLOv3 inference on a webcam
        run(weights='yolov5s.pt', source='0', view_img=True)
        ```
    """
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model
    device = select_device(device)
    names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]  # class names
    stride = 32
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im)
        im, _, _ = letterbox_my(im.permute(1,2,0).numpy(), imgsz, auto=False, stride=32)
        #im = im.astype(np.float32)  # uint8 to fp16/32
        #im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        im = np.transpose(im, (0,3,1,2))
        # Inference
        preds = executor.forward(im)
        preds = [preds[0]*0.2433417, preds[1]*0.3090818]
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

        # NMS
        pred = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
            p = Path(p)  # to Path
            txt_path = os.path.join(save_dir, "labels")
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                save_path = os.path.join(save_dir, p.name)
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
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
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)
                print(f"The image with the result is saved in: {save_path}")


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