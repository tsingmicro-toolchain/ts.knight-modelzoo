import torch
import numpy as np
import glob
import contextlib
import itertools
from tabulate import tabulate
import json
import io
import tempfile
import json
from tqdm import tqdm
from collections import ChainMap, defaultdict
from yolox.data.datasets import COCO_CLASSES
from yolox.data import COCODataset, ValTransform
from yolox.evaluators import COCOEvaluator
from onnx_quantize_tool.common.register import onnx_infer_func, pytorch_model
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh,
    meshgrid
)
IMAGE_SIZE = (640, 640)

def get_eval_dataset(**kwargs):

    testdev = kwargs.get("testdev", False)
    legacy = kwargs.get("legacy", False)
    data_dir = kwargs.get("data_dir", None)
    return COCODataset(
        data_dir=data_dir,
        json_file='instances_val2017.json',
        name="val2017",
        img_size=IMAGE_SIZE,
        # img_size=(416, 416),
        preproc=ValTransform(legacy=legacy),
    )

def get_eval_loader(batch_size, is_distributed, **kwargs):
    valdataset = get_eval_dataset(**kwargs)


    sampler = torch.utils.data.SequentialSampler(valdataset)

    dataloader_kwargs = {
        # "num_workers": self.data_num_workers,
        "pin_memory": True,
        "sampler": sampler,
    }
    dataloader_kwargs["batch_size"] = batch_size
    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    return val_loader

def get_evaluator(batch_size, is_distributed, testdev=False, legacy=False, data_dir=None):
    return COCOEvaluator(
        dataloader=get_eval_loader(batch_size, is_distributed,
                                        testdev=testdev, legacy=legacy, data_dir=data_dir),
        img_size=IMAGE_SIZE,
        # img_size=(416,416),
        confthre=0.001,
        nmsthre=0.65,
        num_classes=80,
        testdev=testdev,
    )
# hw 8400 85 _stride 8 16 32 outputs 1 85 8400
def decode_outputs(outputs, dtype, hw, _stride):
    grids = []
    strides = []
    for (hsize, wsize), stride in zip(hw, _stride):
        yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1).type(dtype)
    strides = torch.cat(strides, dim=1).type(dtype)

    outputs = torch.cat([
        (outputs[..., 0:2] + grids) * strides,
        torch.exp(outputs[..., 2:4]) * strides,
        outputs[..., 4:]
    ], dim=-1)
    return outputs

def convert_to_coco_format(evaluator, outputs, info_imgs, ids, return_outputs=False):
    data_list = []
    image_wise_data = defaultdict(dict)
    for (output, img_h, img_w, img_id) in zip(
        outputs, info_imgs[0], info_imgs[1], ids
    ):
        if output is None:
            continue
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        scale = min(
            evaluator.img_size[0] / float(img_h), evaluator.img_size[1] / float(img_w)
        )
        bboxes /= scale
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        image_wise_data.update({
            int(img_id): {
                "bboxes": [box.numpy().tolist() for box in bboxes],
                "scores": [score.numpy().item() for score in scores],
                "categories": [
                    evaluator.dataloader.dataset.class_ids[int(cls[ind])]
                    for ind in range(bboxes.shape[0])
                ],
            }
        })

        bboxes = xyxy2xywh(bboxes)

        for ind in range(bboxes.shape[0]):
            label = evaluator.dataloader.dataset.class_ids[int(cls[ind])]
            pred_data = {
                "image_id": int(img_id),
                "category_id": label,
                "bbox": bboxes[ind].numpy().tolist(),
                "score": scores[ind].numpy().item(),
                "segmentation": [],
            }  # COCO json format
            data_list.append(pred_data)

    if return_outputs:
        return data_list, image_wise_data
    return data_list
    
def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table

def evaluate_prediction(evaluator, data_dict):


    annType = ["segm", "bbox", "keypoints"]


    info = ""

    # Evaluate the Dt (detection) json comparing with the ground truth
    if len(data_dict) > 0:
        cocoGt = evaluator.dataloader.dataset.coco
        # TODO: since pycocotools can't process dict in py36, write data to json file.
        if evaluator.testdev:
            json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
            cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
        else:
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
        try:
            from yolox.layers import COCOeval_opt as COCOeval
        except ImportError:
            from pycocotools.cocoeval import COCOeval


        cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
        cocoEval.evaluate()
        cocoEval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        info += redirect_string.getvalue()
        cat_ids = list(cocoGt.cats.keys())
        cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
        if evaluator.per_class_AP:
            AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
            info += "per class AP:\n" + AP_table + "\n"
        if evaluator.per_class_AR:
            AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
            info += "per class AR:\n" + AR_table + "\n"
        return cocoEval.stats[0], cocoEval.stats[1], info
    else:
        return 0, 0, info
# deprecated
@pytorch_model.register("yolox_s")
def yolox_s(weight_path=None):
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
    depth = 0.33
    width = 0.50
    num_classes =80
    act ="silu"
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eps =1e-3
                m.momentum = 0.03
    in_channels=[256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act=act)
    model = YOLOX(backbone, head)
    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    if weight_path:
        ckpt = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(ckpt["model"])
    in_dict ={
    "model": model,
    "inputs":[torch.randn((1,3,IMAGE_SIZE[0],IMAGE_SIZE[1]))]
    }
    return in_dict


# 注意coco 数据集 t 416 s/m 640
@onnx_infer_func.register("infer_yolox_small")
def infer_yolox_small(executor):
    batch_size = executor.batch_size
    iteration = executor.iteration
    data_dir = executor.dataset
    num_classes = 80
    confthre = 0.001
    nmsthre = 0.65
    evaluator = get_evaluator(batch_size, False, False, False, data_dir)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True
    ids = []
    data_list = []
    output_data = defaultdict()
    progress_bar = tqdm
    strides = [8, 16, 32]
    inference_time = 0
    nms_time = 0
    n_samples = max(len(evaluator.dataloader) - 1, 1)

    for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(evaluator.dataloader)):
        imgs *= 255
        imgs = imgs.numpy().astype(np.uint8)
        outputs = executor.forward(imgs)
        for i in range(len(outputs)):
            outputs[i] = torch.from_numpy(outputs[i])
        hw = [x.shape[-2:] for x in outputs]
        # # [batch, n_anchors_all, 85]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in outputs], dim=2
        ).permute(0, 2, 1)
        outputs = decode_outputs(outputs, torch.FloatTensor, hw, strides)
        outputs = postprocess(
            outputs, num_classes, confthre, nmsthre
        )
        data_list_elem, image_wise_data = convert_to_coco_format(evaluator,
            outputs, info_imgs, ids, return_outputs=True)
        data_list.extend(data_list_elem)
        output_data.update(image_wise_data)
        # break
        if cur_iter > iteration:
            break

    eval_results = evaluate_prediction(evaluator, data_list)
    print(eval_results)
    return 1

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis
import cv2, os

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

@onnx_infer_func.register("infer_yolox_small_plot")
def infer_yolox_small_plot(executor):
    image_path = executor.dataset
    input_shape = IMAGE_SIZE
    origin_img = cv2.imread(image_path)
    img, ratio = preprocess(origin_img, input_shape)
    img = img.astype(np.uint8)
    img = img[None, :, :, :]
    img.flatten().tofile(os.path.join(executor.save_dir, "model_input.bin"))
    output = executor.forward(img)
    write_numpy_to_file(output[0], os.path.join(executor.save_dir, "817.txt"))
    write_numpy_to_file(output[1], os.path.join(executor.save_dir, "843.txt"))
    write_numpy_to_file(output[2], os.path.join(executor.save_dir, "869.txt"))
    print(f'\nsave result to {executor.save_dir}')
    output[0] = torch.from_numpy(output[0])*0.02798874
    output[1] = torch.from_numpy(output[1])*0.02818405
    output[2] = torch.from_numpy(output[2])*0.02513285
    output = torch.cat([x.flatten(start_dim=2) for x in output], dim=2).permute(0, 2, 1)
    
    predictions = demo_postprocess(output.numpy(), input_shape)[0]

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
    conf = 0.9
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        for i in range(len(final_boxes)):
            box = final_boxes[i]
            score = final_scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            print("x0,y0,x1,y1 ", x0,y0,x1,y1)
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=conf, class_names=COCO_CLASSES)

    output_path = os.path.abspath(os.path.join(executor.save_dir, os.path.basename(image_path)))
    print(f'save picture to {output_path}')
    cv2.imwrite(output_path, origin_img)

