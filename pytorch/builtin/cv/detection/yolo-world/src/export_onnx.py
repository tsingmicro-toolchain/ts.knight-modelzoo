# -*- coding: UTF-8 -*-
"""
@Project ：YOLO-World 
@IDE     ：PyCharm 
@Author  ：gxs
@Date    ：2025/4/2 星期三 15:38 
"""
import os
from typing import List, Sequence, Tuple
import struct
import time
import itertools
from transformers import AutoTokenizer
import numpy as np
import cv2
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor
import onnx
import onnxsim
import onnxruntime
from mmengine.config import Config
from mmdet.apis import init_detector
from yolo_world.models.dense_heads import YOLOWorldHead, YOLOWorldHeadModule
from yolo_world.models.detectors import YOLOWorldDetector
from yolo_world.models.backbones import MultiModalYOLOBackbone, HuggingCLIPLanguageBackbone


class DFL(nn.Module):
    """
    NOTE: 为了转成onnx时候输出是我想要的格式，DFL代码部分使用卷积替换源代码中的矩阵乘
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
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class MultiModalYOLOBackboneNEW(MultiModalYOLOBackbone):
    def __init__(self):
        pass

    def forward_text(self, input_ids, attention_mask):
        print("run MultiModalYOLOBackboneNEW")
        assert self.with_text_model, "forward_text() requires a text model"
        txt_feats = self.text_model(input_ids, attention_mask)
        return txt_feats


class YOLOWorldHeadModuleNEW(YOLOWorldHeadModule):
    def __init__(self):
        pass

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        print("run YOLOWorldHeadModuleNEW")
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        bbox_dist_preds = reg_pred(img_feat)

        ### ========================================================================
        bbox_dist_preds = bbox_dist_preds.reshape([b, -1, h * w])
        # (b, 4, h*w)
        bbox_preds = self.dfl(bbox_dist_preds)
        bbox_preds = bbox_preds.reshape(b, -1, h, w)
        # cls_logit = cls_logit.sigmoid()
        ### ========================================================================

        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


class YOLOWorldHeadNEW(YOLOWorldHead):
    def __init__(self, ) -> None:
        pass

    def forward(self, img_feats, txt_feats):
        print("run YOLOWorldHeadNEW")
        """Forward features from the upstream network."""
        cls_preds, bbox_preds = self.head_module(img_feats, txt_feats)
        outputs = []
        for cls, box in zip(cls_preds, bbox_preds):
            cls = cls.sigmoid()
            reduce_max, _ = torch.max(cls, dim=1, keepdim=True)
            outputs.append(torch.cat([box, reduce_max, cls], dim=1))

        return outputs


class YOLOWorldDetectorNEW(YOLOWorldDetector):
    def __init__(self):
        pass

    def forward(self, images, txt_feats):
        print("run YOLOWorldDetectorNEW")
        return self._forward(images, txt_feats)

    def _forward(self, images, txt_feats):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(images, txt_feats)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(self, images: Tensor, txt_feats) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # # NOTE:判断images的数据类型
        # if images.dtype == torch.uint8:
        #     images = images.to(torch.float32)
        # 需要添加上BN
        # TODO:初始化一个BN，用来计算检测模型归一化这个计算
        # input_channels = images.shape[1]
        # mean = (0., 0., 0.)
        # std = (255., 255., 255.)
        # new_bn = torch.nn.BatchNorm2d(input_channels)
        # new_bn.weight = nn.Parameter(torch.ones((input_channels,)))
        # new_bn.bias = nn.Parameter(torch.zeros((input_channels,)))
        # new_bn.running_mean = torch.zeros((input_channels,)) + torch.FloatTensor(mean)
        # new_bn.running_var = torch.ones((input_channels,)) * torch.FloatTensor(std) * torch.FloatTensor(std)
        # new_bn.eval()
        
        # images = new_bn(images)

        img_feats = self.backbone.forward_image(images)
        # txt_feats = self.backbone.forward_text(input_ids, attention_mask)

        img_feats = self.neck(img_feats, txt_feats)

        return img_feats, txt_feats


def read_bin_file_to_float(file_path):
    with open(file_path, "rb") as rbF:
        _array = list()
        while True:
            binary_data = rbF.read(4)
            # 如果没有更多数据可读，则跳出循环
            if not binary_data:
                break
            # 使用struct.unpack将二进制数据解包为浮点数
            num = struct.unpack('f', binary_data)[0]
            _array.append(num)
    return _array


if __name__ == '__main__':

    config = "src/configs/finetune_coco/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py"
    checkpoint = "weight/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_ep80-492dc329.pth"

    cfg = Config.fromfile(config)
    model = init_detector(cfg, checkpoint=checkpoint, device="cpu", palette="coco")

    model_new = YOLOWorldDetectorNEW()
    model_new.__dict__.update(model.__dict__)
    model = model_new

    backbone_old = model.backbone
    backbone_new = MultiModalYOLOBackboneNEW()
    backbone_new.__dict__.update(backbone_old.__dict__)
    model.backbone = backbone_new

    bbox_head_old = model.bbox_head
    bbox_head_new = YOLOWorldHeadNEW()
    bbox_head_new.__dict__.update(bbox_head_old.__dict__)
    model.bbox_head = bbox_head_new

    head_model_old = model.bbox_head.head_module
    head_model_new = YOLOWorldHeadModuleNEW()
    head_model_new.__dict__.update(head_model_old.__dict__)
    ### =================
    dfl = DFL(c1=16)
    if not hasattr(head_model_new, "dfl"):
        setattr(head_model_new, "dfl", dfl)
    ### =================
    model.bbox_head.head_module = head_model_new

    # print(model)
    model.eval()

    time_str = time.strftime("%y%m%d%H%M%S")
    onnx_path = os.path.join("./weight", "yolo-world-v2-s-image.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # =======================================
    '''
    # image = cv2.imread("testImg.jpg")  # (384, 640)
    image = cv2.imread("testImg640.jpg")  # (640, 640)
    image = image[..., ::-1]  # BGR -> RGB
    image = image.transpose(2, 0, 1)  # hwc -> chw
    test_in = torch.from_numpy(image[None, ...].astype(dtype=np.float32))

    nc = 7
    text_feats = torch.tensor(read_bin_file_to_float("names7-norm-nch.bin"),
                              dtype=torch.float32).reshape((nc, 512))
    # text_feats = torch.tensor(read_bin_file_to_float("names19-20250415-norm-nch.bin"), dtype=torch.float32).reshape((19, 512))
    num_per_batch = text_feats.shape[0]
    # NOTE:因为Knight不支持直接调用的L2-norm算子，需要手动实现
    # 计算步骤：平方，求和，开方
    # text_l2_norm = torch.sqrt(torch.sum(torch.pow(text_feats, 2), dim=-1, keepdim=True))
    # text_feats = text_feats / text_l2_norm
    text_feats = text_feats.reshape(-1, num_per_batch, text_feats.shape[-1])
    text_feats = text_feats.detach().numpy()
    
    # ================================================
    '''
    test_in = np.random.randn(1,3,640,640).astype(np.float32)
    text_feats = np.random.randn(1,80,512).astype(np.float32)
    torch.onnx.export(
        model,
        (torch.from_numpy(test_in), torch.from_numpy(text_feats)),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['images', "text"],
        output_names=['out0', "out1", "out2"],
        verbose=False,
    )

    # ==========================================================================================
    print("load onnx")
    onnx_model = onnx.load(onnx_path)
    print("check onnx model")
    onnx.checker.check_model(onnx_model)
    print("simplify onnx model")
    onnx_model, check = onnxsim.simplify(onnx_model)

    print("save onnx model")
    onnx.save(onnx_model, onnx_path)
    # ==========================================================================================

    model = onnx_model

    graph = model.graph

    print("\n\n# ==========================================================================================")

    print("Model Inputs:")
    for i, input_tensor in enumerate(graph.input):
        name = input_tensor.name
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        data_type = input_tensor.type.tensor_type.elem_type
        print(f"  Input {i}:")
        print(f"    Name: {name}")
        print(f"    Shape: {shape}")
        print(f"    Data Type: {data_type}")

    print("\nModel Outputs:")
    for i, output_tensor in enumerate(graph.output):
        name = output_tensor.name
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        data_type = output_tensor.type.tensor_type.elem_type
        print(f"  Output {i}:")
        print(f"    Name: {name}")
        print(f"    Shape: {shape}")
        print(f"    Data Type: {data_type}")
    print("\n\n# ==========================================================================================")
    # ==========================================================================================
