import paddle
from paddle.vision.models import resnet18 as Resnet18
from frontends.paddle.model.ghostnet import GhostNet_x1_0
from frontends.paddle.model.densenet import DenseNet121
from frontends.paddle.model.alexnet import AlexNet
from frontends.paddle.model.mobilenet_v2 import MobileNetV2
from frontends.paddle.model.shufflenet_v2 import ShuffleNetV2_x1_0
from frontends.paddle.model.squeezenet import SqueezeNet1_1
from frontends.paddle.model.googlenet import GoogLeNet
from frontends.paddle.model.resnext import ResNeXt50_32x4d

def alexnet(weight=None):
    model = AlexNet()
    if weight:
        model.set_dict(paddle.load(weight))
    return model, ["batch_size", 3, 224, 224]

def mobilenet_v2(weight=None):
    model = MobileNetV2()
    if weight:
        model.set_dict(paddle.load(weight))
    return model, ["batch_size", 3, 224, 224]

def ghostnet(weight=None):
    model = GhostNet_x1_0()
    if weight:
        model.set_dict(paddle.load(weight))
    return model, ["batch_size", 3, 224, 224]

def shufflenet(weight=None):
    model = ShuffleNetV2_x1_0()
    if weight:
        model.set_dict(paddle.load(weight))
    return model, ["batch_size", 3, 224, 224]

def resnet18(weight=None):
    model = Resnet18()
    if weight:
        model.set_dict(paddle.load(weight))
    return model, ["batch_size", 3, 224, 224]

def squeezenet(weight=None):
    model = SqueezeNet1_1()
    if weight:
        model.set_dict(paddle.load(weight))
    return model, ["batch_size", 3, 224, 224]

def densenet(weight=None):
    model = DenseNet121()
    if weight:
        model.set_dict(paddle.load(weight))
    return model, ["batch_size", 3, 224, 224]

def resnext50(weight=None):
    model = ResNeXt50_32x4d()
    if weight:
        model.set_dict(paddle.load(weight))
    return model, ["batch_size", 3, 224, 224]

def googlenet(weight=None):
    model = GoogLeNet()
    if weight:
        model.set_dict(paddle.load(weight))
    return model, ["batch_size", 3, 224, 224]
