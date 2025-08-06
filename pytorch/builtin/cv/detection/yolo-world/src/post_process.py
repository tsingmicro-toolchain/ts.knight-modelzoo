# -*- coding: UTF-8 -*-
"""
@Project yoloworld 
@IDE     PyCharm 
@Date    2025/7/17  

"""
import os
import csv
from typing import List
import argparse
import numpy as np
import cv2
import torch
import torchvision

from onnx_quantize_tool.common.register import onnx_infer_func

ARGS_DICT = dict(
        save_root= '/TS-KnightDemo/Samples/output/yoloworld/data',
        class_name_path = '/ts.knight-modelzoo/pytorch/builtin/cv/detection/yolo-world/data/coco_names.txt',
        text_feats_path = '/ts.knight-modelzoo/pytorch/builtin/cv/detection/yolo-world/data/texts/coco80.npy',
        # NOTE: 保存结果的类型  0:不保存任何东西，只做检测[default];
        #                                          1:保存onnx三个输出为txt，为C++提供输入；(部门内部使用);
        #                                          2:只保存交付需要的中间变量；(交付使用);
        #                                          3:1,2的都保存;
        save_flag=3,
        # NMS的置信度阈值
        conf_thres=0.5,
        # NMS的iou阈值
        iou_thres=0.45,
        kpt_v_thres=0.5,
        kpt_shape=(0, 3),
        scale_outputs=(0.05439655, 0.05514111, 0.05740907),
        input_chw=(3, 640, 640),  # 360P 
        max_det=300,
    )

def get_color(index):
    """
    输入一个索引获取一个特定的颜色
    @param index:
    @return:
    """
    index = int(index) + 4
    # 计算 RGB 分量
    r = (index * 67) % 256  # 用某种公式对红色分量进行计算
    g = (index * 123) % 256  # 用某种公式对绿色分量进行计算
    b = (index * 89) % 256  # 用某种公式对蓝色分量进行计算

    return r, g, b


def xyxy2xywh(x):
    """
    Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    @param x:
    @return:
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    @param x:
    @return:
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction,
                        nc,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        kpt_shape=(0, 2),
                        max_det=300,
                        multi_label=False,
                        ):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    @param prediction: [xywh conf nc*score kpt_num]
    @param nc: 类别数量
    @param conf_thres: 最后检测的置信度阈值
    @param iou_thres: nms使用iou阈值
    @param kpt_shape: 关键点的维度, 关键点的个数&一个关键点维度
    @param max_det: 一张图片最多检测数量
    @param multi_label: 一个框多个标签
    @return: list of detections, on (n,6+kpt[0]*kpt[1]) tensor per image [xyxy, conf, cls, kpt...]
    """
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    bs = prediction.shape[0]  # batch size
    kpt_num = kpt_shape[0] * kpt_shape[1]  # 关键点的个数
    # 从该索引之后就是关键点的信息
    mi = 4 + nc  # mask start index
    # 置信度大于阈值的(b, ...)
    xc = prediction[..., 4:mi].amax(dim=2) > conf_thres  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    multi_label &= nc > 1  # 一个框有多个类别
    # 输出的结果
    output = [torch.zeros((0, 6 + kpt_num), device=device)] * bs
    # 单张图片处理
    for xi, x in enumerate(prediction):  # image index, image inference
        # 使用置信度挑选出大于置信度阈值的框
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Box/Mask
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # zero columns if no masks
        mask = x[:, mi:]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 4:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 4:mi].max(1, keepdim=True)  # 求所有类别分数的最大值和随影的索引
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue

        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        # classes 让类别差异变得更大
        c = x[:, 5:6] * max_wh
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]

        # NMS
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        # limit detections
        i = i[:max_det]
        # 结果保存起来
        output[xi] = x[i]

    return output


class V8PostProcess:
    def __init__(self, nc, kpt_shape):
        self.nc = nc
        self.kpt_shape = kpt_shape
        self.no = 4 + nc + kpt_shape[0] * kpt_shape[1]
        self.nl = 3
        # 因为是anchor-free的算法没有anchor，只有锚点，为了和anchor-Base的代码复用，定义anchor-free的个数为1
        self.na = 1

        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid

        # 网络输出层的步距
        self.stride = torch.Tensor([8, 16, 32])

    def post_process(self, x, xywh=True):
        """
        输入的格式是[ltrb nc*score kpt...]
        @param x: 网络的三个输出层
        @param xywh:
        @return:
        """
        assert len(x) == self.nl, "ERROR"
        grid_cell_offset = 0.5  # 默认不变
        z = []

        kpt_num = self.kpt_shape[0] * self.kpt_shape[1]
        kpt_stride = self.kpt_shape[1]
        kpt_start_index = 4 + self.nc

        for i in range(len(x)):
            bs, _, ny, nx = x[i].shape
            stride = self.stride[i]

            # (b, na * no, h, w) -> (b, na, no, h, w) -> (b, na, h, w, no)
            out = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # 没有锚点的时候，先生成锚点
            if self.grid[i].shape[2:4] != out.shape[2:4]:
                # grid: (1, na, h, w, xy)
                self.grid[i] = self._make_grid(nx, ny, grid_cell_offset=grid_cell_offset)

            # (b, 1, h, w, 2)
            grid = self.grid[i]  # (x, y)

            ### ==================================
            # 将预测的坐标映射回来
            lt = out[..., 0:2]  # (b, 1, h, w, 2)
            rb = out[..., 2:4]  # (b, 1, h, w, 2)

            # 最后预测对应输入图的边界框
            x1y1 = (grid - lt) * stride
            x2y2 = (grid + rb) * stride

            if kpt_num != 0:
                ### ==================================
                # 关键点
                kpts_x = out[..., kpt_start_index::kpt_stride]
                kpts_y = out[..., kpt_start_index + 1::kpt_stride]

                kpts_x = (kpts_x * 2.0 + (grid[..., 0:1] - 0.5)) * stride  # x
                kpts_y = (kpts_y * 2.0 + (grid[..., 1:2] - 0.5)) * stride  # y
                ### ==================================

            if xywh:
                c_xy = (x1y1 + x2y2) / 2
                wh = x2y2 - x1y1

                out[..., 0:2] = c_xy
                out[..., 2:4] = wh

                if kpt_num != 0:
                    out[..., kpt_start_index::kpt_stride] = kpts_x
                    out[..., kpt_start_index + 1::kpt_stride] = kpts_y

                # (b, na, h, w, no) -> (b, na*h*w, no)
                z.append(out.view(bs, -1, self.no))

            else:
                out[..., 0:2] = x1y1
                out[..., 2:4] = x2y2

                if kpt_num != 0:
                    out[..., kpt_start_index::kpt_stride] = kpts_x
                    out[..., kpt_start_index + 1::kpt_stride] = kpts_y

                # (b, na, h, w, no) -> (b, na*h*w, no)
                z.append(out.view(bs, -1, self.no))

            # (b, na*(h1*w1 + h2*w2 +h3*w3), no)
        return torch.cat(z, dim=1)

    def _make_grid(self, nx, ny, grid_cell_offset=0.5, dtype=torch.float32):
        # 锚点的shape
        shape = 1, self.na, ny, nx, 2

        sx = torch.arange(end=nx, dtype=dtype)  # shift x
        sy = torch.arange(end=ny, dtype=dtype)  # shift y

        # torch老版本默认就是ij格式
        try:
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')  # torch>=10
        except Exception as _:
            sy, sx = torch.meshgrid(sy, sx)  # torch>=0.7

        # (nx, ny, 2)
        grid = torch.stack((sx, sy), -1) + grid_cell_offset
        # (ny, nx, 2) -> (1, 1, ny, nx, 2)
        grid = grid.expand(shape)

        return grid


def pre_process_resize_img(image, new_shape_wh):
    """
    宽高等比例缩放
    @param image:
    @param new_shape_wh: (target_width, target_height)
    @return:
    """
    # 获取原始图像的宽度和高度
    height, width, chanel = image.shape

    # 目标宽度和高度
    target_width, target_height = new_shape_wh

    # 计算宽度和高度的缩放比例
    ratio = min((1. * target_width) / width, (1. * target_height) / height)

    # 计算缩放后的宽度和高度
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # 等比例缩放图像
    resized_img = cv2.resize(image, (new_width, new_height))

    # 计算需要填充的宽度和高度
    dw = (target_width - new_width) // 2
    dh = (target_height - new_height) // 2

    # 创建一个白色底图
    padded_img = np.zeros(shape=[target_height, target_width, chanel], dtype=np.uint8)

    # 将缩放后的图像放置在中心位置
    padded_img[dh:dh + new_height, dw:dw + new_width, :] = resized_img

    return padded_img, ratio, dw, dh


def post_scale_boxes(boxes, ratio, dw, dh, img_hw):
    """
    和上面的pre_process_resize_img配套使用，将模型输出的结果映射回原图大小
    :param boxes:
    :param ratio:
    :param dw:
    :param dh:
    :param img_hw: 用来把box限制到图片内
    :return:
    """
    # boxes：输入的检测框，格式为 [xmin, ymin, xmax, ymax]

    # 将检测框进行平移，以适应填充后的图像
    boxes_translated = boxes - np.array([dw, dh, dw, dh])
    # 将检测框进行等比例缩放，已经恢复到原图尺寸
    boxes_scaled = boxes_translated / ratio

    # 限制在图片内
    boxes_scaled[..., 0::2] = np.clip(boxes_scaled[..., 0::2], 0, img_hw[1])
    boxes_scaled[..., 1::2] = np.clip(boxes_scaled[..., 1::2], 0, img_hw[0])

    return boxes_scaled


def draw_box(image, boxes, names=None):
    """

    :param image:
    :param boxes: [n, 6]
    :param names:
    :return:
    """
    # 自适应图片大小计算线宽和文字大小
    # 图片大小
    img_h, img_w = image.shape[:2]
    # box线宽
    box_thickness = max(round(sum((img_h, img_w)) / 2 * 0.003), 2)
    # font_size = max(round(sum((img_h, img_w)) / 2 * 0.035), 12)
    # text线宽
    text_thickness = max(box_thickness - 1, 1)
    # text字体大小
    fontScale = box_thickness / 4.
    # 字体
    fontFace = 0

    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cls_id = int(cls_id)

        cv2.rectangle(image, pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=get_color(cls_id),
                      thickness=box_thickness, lineType=cv2.LINE_AA)


        # 在框上显示的text
        if names is None:
            text = f"{cls_id}:{conf:.2f}"
        else:
            text = f"{names[cls_id]}:{conf:.2f}"

        # 用于计算特定文本字符串在给定字体和大小下的尺寸
        text_w, text_h = cv2.getTextSize(text=text,
                                         fontFace=fontFace,
                                         fontScale=fontScale,
                                         thickness=text_thickness)[0]

        # 防止写的text超过了上边界，导致看不到，如果能写外面就写外面
        inside_x = (img_w - x1 - text_w) >= 5  # 如果横着会导致文本超过图像边界，text向左挪一挪
        inside_y = (y1 - text_h) >= 8
        # 计算字体左下角坐标
        text_l = x1 if inside_x else x1 - (x1 + text_w + 3 - img_w)
        text_b = (y1 - 4) if inside_y else y1 + text_h + 5

        # 写文字
        cv2.rectangle(image,
                      pt1=(text_l, text_b - text_h),
                      pt2=(text_l + text_w, text_b + 4),
                      color=(255, 255, 255), thickness=-1)
        cv2.putText(image, text=text,
                    org=(text_l, text_b),  # 防止超过上边界
                    fontFace=fontFace, fontScale=fontScale, thickness=text_thickness,
                    color=get_color(cls_id))

    return image

class SaveTempInfo:
    """
    保存推理过程中的缓存变量
    """

    def __init__(self, save_root):
        self.save_root = save_root

    def save_cpp_info(self, file_name: str,
                      org_img_path,
                      ratio, dw, dh,
                      outputs: List[torch.Tensor]):
        """
        保存的是在C++程序中使用的模型输出的结果
        没有经过反量化系数
            model_outputs: C++读取的数据txt目录
                file_name:每个图片的名字一个目录
                    H.txt: 图片的三个输出都会保存起来
                    meta.txt: 把原图resize到输入图大小时的缩放比例和dw/dh
        @param file_name:
        @param org_img_path:
        @param ratio:
        @param dw:
        @param dh:
        @param outputs:
        @return:
        """
        model_outputs_root = os.path.join(self.save_root, "model_outputs", file_name)
        os.makedirs(model_outputs_root, exist_ok=True)

        # 保存Meta信息
        if not os.path.exists(os.path.join(model_outputs_root, "meta.txt")):
            # 保存前处理信息
            ratios = (ratio, ratio)  # (ry, rx)
            pad = (dw, dh)  # (dw, dh)

            with open(os.path.join(model_outputs_root, "meta.txt"), "w") as wFile:
                wFile.writelines(f"{org_img_path}\n")  # image path

                wFile.writelines(f"{ratios[0]}\n")  # ratio y
                wFile.writelines(f"{ratios[1]}\n")  # ratio x

                wFile.writelines(f"{pad[0]}\n")  # dw
                wFile.writelines(f"{pad[1]}\n")  # dh

        # 模型推理的结果
        for output in outputs:
            # 因为这个脚本中都是一张图片一张图片的加载，所以B=1
            B, C, H, W = output.shape
            # save feature map
            with open(os.path.join(model_outputs_root, f"{H}.txt"), "w") as wFile:
                for c in range(C):
                    for h in range(H):
                        for w in range(W):
                            value = output[0, c, h, w]
                            wFile.writelines(f"{value:.10f}\n")

    

    def save_post(self, file_name, outputs: torch.Tensor):
        """
        后处理后的结果
            经过后处理类之后的数据集结果
        @param file_name:
        @param outputs:
        @return:
        """
        save_post = os.path.join(self.save_root, "post")
        os.makedirs(save_post, exist_ok=True)
        temp_outputs = outputs.clone().view(-1)
        with open(os.path.join(save_post, f"{file_name}_post.txt"), 'w', encoding="utf-8") as wFile:
            for v in temp_outputs:
                wFile.write(f"{v:.6f}\n")

    def save_nms(self, file_name, outputs: np.ndarray, img_hw, kpt_shape):
        """
        nms后的结果保存，对应模型输入图的结果
            经过NMS之后的结果
        @param file_name:
        @param outputs: xyxy conf cls kpt...
        @param img_hw:
        @param kpt_shape:
        @return:
        """
        kpt_num = kpt_shape[0] * kpt_shape[1]
        kpt_stride = kpt_shape[1]
        target_height, target_width = img_hw

        save_nms = os.path.join(self.save_root, "nms")
        os.makedirs(save_nms, exist_ok=True)

        temp_outputs = outputs.copy()
        with open(os.path.join(save_nms, f"{file_name}_nms.txt"), 'w', encoding="utf-8") as wFile:
            wFile.write(f"{'=' * 10}\n")
            wFile.write(f"NMS后的结果\n")
            wFile.write(f"对应输入图像高宽h:{target_height} w:{target_width}\n")
            wFile.write(f"关键点的shape: {kpt_shape}\n")
            if kpt_num == 0:
                wFile.write(f"cls_id x1 y1 x2 y2 conf\n")
            else:
                if kpt_shape[1] == 2:
                    wFile.write(f"cls_id x1 y1 x2 y2 conf xyxy...\n")
                elif kpt_shape[1] == 3:
                    wFile.write(f"cls_id x1 y1 x2 y2 conf xyvxyv...\n")
            wFile.write(f"绝对输出\n")
            wFile.write(f"{'=' * 10}\n")

            for output in temp_outputs:
                xyxy_str = " ".join([f"{v:.2f}" for v in output[:4]])
                conf = output[4]
                cls = output[5]
                # NOTE:如果没有kpt可以执行，要注意结果好像并不是空字符串，
                kpts_str = " ".join([f"{v:.2f}" for v in output[6:]])
                if kpt_num:
                    wFile.write(f"{int(cls)} {xyxy_str} {conf:.4f} {kpts_str}\n")
                else:
                    wFile.write(f"{int(cls)} {xyxy_str} {conf:.4f}\n")

            wFile.write(f"{'=' * 10}\n")
            wFile.write(f"归一化后输出\n")
            wFile.write(f"{'=' * 10}\n")
            for output in temp_outputs:
                output[..., 0:4:2] /= target_width
                output[..., 1:4:2] /= target_height

                if kpt_num != 0:  # 存在关键点
                    output[..., 6::kpt_stride] /= target_width
                    output[..., 7::kpt_stride] /= target_height

                xyxy_str = " ".join([f"{v:.6f}" for v in output[:4]])
                conf = output[4]
                cls = output[5]
                # NOTE:如果没有kpt可以执行，要注意结果好像并不是空字符串，
                kpts_str = " ".join([f"{v:.6f}" for v in output[6:]])
                if kpt_num:
                    wFile.write(f"{int(cls)} {xyxy_str} {conf:.4f} {kpts_str}\n")
                else:
                    wFile.write(f"{int(cls)} {xyxy_str} {conf:.4f}\n")

    def save_result_xywhn(self, file_name: str, outputs: np.ndarray, img_bgr_hw, kpt_shape):
        """
        保存最后的结果，对应原图大小的归一化之后的结果，
        保存顺序是:[cls box conf kpt]
        @param file_name:
        @param outputs: xyxy conf cls kpt...
        @param img_bgr_hw:
        @param kpt_shape:
        @return:
        """
        # 保存最后的结果
        save_txt_root = os.path.join(self.save_root, "txt_xywhn")
        os.makedirs(save_txt_root, exist_ok=True)

        img_h, img_w = img_bgr_hw
        kpt_stride = kpt_shape[1]
        kpt_num = kpt_shape[0] * kpt_shape[1]
        # xyxy conf cls kpt...
        temp_outputs = outputs.copy()

        # xyxy -> xywh
        temp_outputs[..., 0] = (outputs[..., 0] + outputs[..., 2]) / 2.
        temp_outputs[..., 1] = (outputs[..., 1] + outputs[..., 3]) / 2.
        temp_outputs[..., 2] = outputs[..., 2] - outputs[..., 0]
        temp_outputs[..., 3] = outputs[..., 3] - outputs[..., 1]
        # NOTE: 归一化
        temp_outputs[..., 0:4:2] /= img_w
        temp_outputs[..., 1:4:2] /= img_h
        with open(os.path.join(save_txt_root, f"{file_name}.txt"), "w", encoding="utf-8") as wFile:
            for temp_out in temp_outputs:
                box = [f"{v:.6f}" for v in temp_out[:4]]
                conf = f"{temp_out[4]:.3f}"
                cls = f"{int(temp_out[5])}"
                # 置信度v小数点后4为，坐标小数点后2位
                kpts = [f"{v:.6f}" for i, v in enumerate(temp_out[6:])]
                _t = [cls, *box, conf, *kpts]
                wFile.write(f"{' '.join(_t)}\n")

    def save_result_xyxy(self, file_name: str, outputs: np.ndarray, kpt_shape):
        """
        保存最后的结果，对应原图的结果
        保存顺序是:[conf box cls kpt]
        @param file_name:
        @param outputs:
        @param kpt_shape:
        @return:
        """
        # 保存最后的结果
        save_txt_root = os.path.join(self.save_root, "txt_xyxy")
        os.makedirs(save_txt_root, exist_ok=True)
        #
        kpt_num = kpt_shape[0] * kpt_shape[1]

        # xyxy conf cls kpt...
        temp_outputs = outputs.copy()
        
        with open(os.path.join(save_txt_root, f"{file_name}.txt"), "w", encoding="utf-8") as wFile:
            i=0
            for temp_out in temp_outputs:
                box = [int(v) for v in temp_out[:4]]
                conf = f"{temp_out[4]:.3f}"
                cls = f"{int(temp_out[5])}"
                # 置信度v小数点后4为，坐标小数点后2位
                kpts = [f"{v:.2f}" for i, v in enumerate(temp_out[6:])]
                print(f'draw box {i}: x1:{box[0]}, y1:{box[1]}, w:{box[2]-box[0]}, h:{box[3]-box[1]}, conf:{conf}, class_id:{cls}')
                i=i+1
                box = [f"{v:.0f}" for v in temp_out[:4]]
                _t = [conf, *box, cls]  
                wFile.write(f"{' '.join(_t)}\n")


class BaseModel:
    def __call__(self, image, text_feats):
        raise NotImplementedError()


class RunModel:
    def __init__(self,
                 model: BaseModel,
                 names,
                 input_chw,
                 kpt_shape,
                 text_feats,
                 scale_outputs=(1., 1., 1.),
                 ):
        self.model = model

        self.nc = len(names)
        self.names = names
        self.text_feats = text_feats
        self.input_chw = input_chw
        self.scale_outputs = scale_outputs
        self.kpt_shape = kpt_shape
        self.post_process = V8PostProcess(nc=self.nc,
                                          kpt_shape=kpt_shape)


    def load_image(self,image_path,outpath):
        # 解析图片名称
        file = os.path.basename(image_path)
        image_bgr = cv2.imread(image_path)

        # 处理图片成输入尺寸
        target_height, target_width = (640, 640)
        # 前处理  BGR -> RGB && resize
        img, ratio, dw, dh = pre_process_resize_img(image_bgr.copy()[..., ::-1],
                                                    new_shape_wh=(target_width, target_height))

        # (h, w, c) -> (c, h, w);
        img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(img)  # 数据连续
        img = img.astype(np.uint8)
        img = img[np.newaxis, :]
        if outpath:
            img.flatten().tofile(os.path.join(outpath, 'model_input.bin'))

        return img

    def detection_post_process(self,savedir,outputs,output_dtype=None,test_data=None):
        ###############后处理
        if outputs[0].shape[1] == 85:
            outputs = [np.concatenate([out[:, 0:4, :, :], out[:, 5:, :, :]], axis=1) for out in outputs]

        # NOTE: 确保后面出来的时候是Tensor, 并且有反量化系数时乘上反量化系数
        for i in range(len(outputs)):
            outputs[i] = torch.from_numpy(outputs[i]) * ARGS_DICT['scale_outputs'][i]    
        # names
        class_name_path = os.path.abspath(ARGS_DICT["class_name_path"])
        with open(class_name_path, "r") as rFile:
            names = [line.strip() for line in rFile.readlines()]
        nc = len(names)

        text_feats_path = os.path.abspath(ARGS_DICT["text_feats_path"])

        text_feats = torch.tensor(np.load(text_feats_path), dtype=torch.float32).reshape(nc, 512)
        num_per_batch = text_feats.shape[0]
        text_feats = text_feats.reshape(-1, num_per_batch, text_feats.shape[-1])
        text_feats = text_feats.detach().numpy()

        outputs =self.post_process.post_process(outputs)
        # NOTE: NMS numpy:(xyxy conf cls_id kpt...) shape:[N, 4+1+1+kpt_num]
        outputs = non_max_suppression(prediction=outputs,
                                    nc=self.nc,
                                    conf_thres=ARGS_DICT['conf_thres'],
                                    iou_thres=ARGS_DICT['iou_thres'],
                                    kpt_shape=self.kpt_shape,
                                    max_det=ARGS_DICT['max_det'],
                                    )[0].detach().cpu().numpy()  

        kpt_num = self.kpt_shape[0] * self.kpt_shape[1]
        image_bgr = cv2.imread(test_data)
        img_bgr_h, img_bgr_w = image_bgr.shape[:2]
        img, ratio, dw, dh = pre_process_resize_img(image_bgr.copy()[..., ::-1],
                                                    new_shape_wh=(ARGS_DICT["input_chw"][1], ARGS_DICT["input_chw"][2]))
        # NOTE: 把最后的预测结果映射回原图
        outputs[:, :4] = post_scale_boxes(boxes=outputs[..., :4],
                                            ratio=ratio, dw=dw, dh=dh,
                                            img_hw=(img_bgr_h, img_bgr_w))  # boxes
        saveInfo = SaveTempInfo(savedir)
        cnt_box=len(outputs)
        print(f'After nms, The boxes number is {cnt_box}')
        print("----------------------------------------")
        saveInfo.save_result_xywhn(file_name="xywhn",
                                   outputs=outputs,
                                   img_bgr_hw=image_bgr.shape[:2],
                                   kpt_shape=self.kpt_shape, )
        saveInfo.save_result_xyxy(file_name="xyxy",
                                  outputs=outputs,
                                  kpt_shape=self.kpt_shape, )
        print("----------------------------------------")

        image_bgr = cv2.imread(test_data)

        # 保存原图,png不会被压缩
        org_img_path = os.path.join(savedir, "image_org", "image_org.png")

        # 处理图片成输入尺寸
        target_height, target_width = (640, 640)
        show_img = image_bgr.copy()
        show_img = draw_box(show_img, boxes=outputs[..., :6], names=self.names)

        # 保存图片
        save_image_root = os.path.join(savedir, "show_img")
        os.makedirs(save_image_root, exist_ok=True)
        save_image_path = os.path.join(save_image_root, "result_img.jpg")
        print(f"the picture is saved at {save_image_path}")
        cv2.imwrite(save_image_path, show_img)
        return outputs



def txt2npy(result_file, result_shape):
    act_list = []
    has_point = 0
    no_point = 0
    with open (result_file, 'r') as act_re:
        lines = csv.reader(act_re)
        for row in lines:
            if len(row) != 0 and "[" not in row[0] and 'SHAPE' not in row[0] and"#LAYER_NAME" not in row[0]:
                act_list.append(row[:-1])
                for x in row[:-1]:
                    if '.' in x:
                        has_point += 1
                    else:
                        no_point += 1
    if has_point > 0 and no_point > 0:
        raise ValueError('result file:{result_file} should only has float type numbers or int type numbers.')

    is_float = False
    if has_point > 0:
        is_float = True

    if is_float:
        act_list = [[float(x) for x in row] for row in act_list]
    else:
        act_list = [[int(x) for x in row] for row in act_list]
    return torch.tensor(act_list).reshape(result_shape), is_float

def detect():
    image_path = opt.image
    outpath = opt.save_dir
    ARGS_DICT["scale"] = opt.scales
    outputs = [np.load(numpy) for numpy in opt.numpys]
    # names
    class_name_path = os.path.abspath(ARGS_DICT["class_name_path"])
    with open(class_name_path, "r") as rFile:
        names = [line.strip() for line in rFile.readlines()]
    nc = len(names)

    text_feats_path = os.path.abspath(ARGS_DICT["text_feats_path"])

    text_feats = torch.tensor(np.load(text_feats_path), dtype=torch.float32).reshape(nc, 512)
    num_per_batch = text_feats.shape[0]
    text_feats = text_feats.reshape(-1, num_per_batch, text_feats.shape[-1])
    text_feats = text_feats.detach().numpy()

    runModel = RunModel(model="",
                names=names,
                input_chw=ARGS_DICT['input_chw'],
                kpt_shape=ARGS_DICT['kpt_shape'],
                text_feats=text_feats,
                scale_outputs=ARGS_DICT['scale_outputs'])

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    runModel.detection_post_process(.outpath,outputs,output_dtype=None,test_data=image_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='original image')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
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