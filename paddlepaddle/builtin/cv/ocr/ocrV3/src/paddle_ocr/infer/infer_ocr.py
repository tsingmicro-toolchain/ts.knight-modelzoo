
import os
import numpy as np
from paddle import Tensor
from torch import cosine_similarity
import torch
from paddle_ocr.infer.db_postprocess import DistillationDBPostProcess
from paddle_ocr.infer.distillation_metric import DistillationMetric
from paddle_ocr.infer.cls_postprocess import ClsPostProcess
from paddle_ocr.infer.rec_postprocess import DistillationCTCLabelDecode, CTCLabelDecode
from paddle_ocr.infer.cls_metric import *
from paddle_ocr.infer.rec_metric import *
from onnx_quantize_tool.common.register import onnx_infer_func

@onnx_infer_func.register("infer_ocr_det_model")
def infer_ocr_det_model(executor):
    import paddle
    iteration = executor.iteration
    # batch_size = executor.batch_size
    """
    data format:
        [[image, ratio, label_det, ignore],...]
        with shapes respectivly as follows
        image: 1x3x736x1312 => 1x3x512x896
        ratio: 1x4
        label_det: 1xDx4x2
        ignore: 1xD
    """
    # det_test_data_100 = paddle.load("~/data/ICDAR2015_det/det_test_500_b1_736x1312.pd")
    # det_test_data_100 = paddle.load("/TS-Knight/Quantize/Onnx/example/data/det_test_500_b1_512x896.pd")
    det_test_data_100 = paddle.load("../onnx_examples/paddle_ocr/example/data/det_b1.pd")
    post_process_class = DistillationDBPostProcess(
        model_name=["Student"],
        key="head_out",
        thresh=0.3,
        box_thresh=0.6,
        max_candidates=1000,
        unclip_ratio=1.5
    )
    eval_class = DistillationMetric(key="Student", base_metric_name="DetMetric", main_indicator="hmean")
    
    val_loader = det_test_data_100

    for i, batch in enumerate(val_loader):
        print(f"batch {i}")
        batch_numpy = [it.numpy() for it in batch]
        input_data = batch_numpy[0]  # image
        preds = executor.forward(input_data)
        preds = {"Student": {"maps": preds[0]}}  # format required
        post_result = post_process_class(preds, batch_numpy[1])
        eval_class(post_result, batch_numpy)
        if i + 1 == iteration:
            break
    metric = eval_class.get_metric()
    """
    metrics {
                 'precision': 0,
                 'recall': 0,
                 'hmean': 0
            }
    """
    print(f"metric: {metric}")
    return metric["hmean"]

@onnx_infer_func.register("infer_ocr_cls_model")
def infer_ocr_cls_model(executor):
    import paddle
    iteration = executor.iteration
    # batch_size = executor.batch_size
    """
    data format:
        [[image, label],...]
        with shapes respectivly as follows
        image: Bx3x48x192
        label: B
    """
    cls_test_30_b1 = paddle.load("../onnx_examples/paddle_ocr/example/data/cls_b1.pd")
    post_process_class = ClsPostProcess(label_list=['0','180'])
    eval_class = ClsMetric(main_indicator="acc")
    
    val_loader = cls_test_30_b1

    for i, batch in enumerate(val_loader):
        print(f"batch {i}")
        batch_numpy = [it.numpy() if isinstance(it, paddle.Tensor) else np.array([it]) for it in batch]
        input_data = batch_numpy[0]  # image
        preds = executor.forward(input_data)
        post_result = post_process_class(preds[0], batch_numpy[1])
        eval_class(post_result, batch_numpy)
        if i + 1 == iteration:
            break
    metric = eval_class.get_metric()
    """
    metrics {
                 'acc': 0.83333
            }
    """
    print(f"metric: {metric}")
    return metric["acc"]

@onnx_infer_func.register("infer_ocr_rec_model")
def infer_ocr_rec_model(executor):
    import paddle
    iteration = executor.iteration
    batch_size = executor.batch_size
    """
    data format:
        [[image, student, teacher, ignoreS, ignoreT],...]
        with shapes respectivly as follows
        image: 128x3x48x320 
        student: 128x25
        teacher: 128x25
        ignoreS: 128
        ignoreT: 128
    """  
    rec_test_data = paddle.load("../onnx_examples/paddle_ocr/example/data/rec_b1.pd")  
    
    post_process_class = CTCLabelDecode(
        character_dict_path="../onnx_examples/paddle_ocr/infer/en_dict.txt",
        use_space_char=True
    )
    eval_class = RecMetric(
        main_indicator="acc",
        ignore_space=False)
    
    val_loader = rec_test_data  # 17x[128,3,48,320] or 67*[31,3,48,320]

    for i, batch in enumerate(val_loader):
        print(f"batch {i}")
        batch_numpy = [it.numpy()[:batch_size] for it in batch]
        input_data = batch_numpy[0]  # image
        preds = executor.forward(input_data)[0]
        # preds = {"Student": {"head_out": preds}}  # distillation format required
        post_result = post_process_class(preds, batch_numpy[1])
        eval_class(post_result, batch_numpy)
        if i + 1 == iteration:
            break
    metric = eval_class.get_metric()
    """
    metrics {
                 'acc': 0, 'norm_edit_dis': 0
            }
    """
    print(f"metric: {metric}")
    return metric["acc"]

