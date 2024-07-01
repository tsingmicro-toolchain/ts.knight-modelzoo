import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models
from onnx_quantize_tool.common.register import pytorch_model

@pytorch_model.register("yolo_nas_s")
def yolo_nas_s(weight_path=None):
    model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
    model.eval()
    in_dict = {
        "model": model,
        "inputs": [torch.randn((1,3,640,640))],
        "concrete_args":{"augment":False,"profile":False,"visualize":False,"val":True}
    }
    return in_dict
