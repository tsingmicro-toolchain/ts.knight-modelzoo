import os
import numpy as np
from PIL import Image
import time
import torch
import torchvision
from timm.utils import accuracy, AverageMeter
from torchvision import datasets, transforms
from torch import nn
from typing import Union, List, Dict, Any, cast
from onnx_quantize_tool.common.register import onnx_infer_func, pytorch_model


@pytorch_model.register("vgg16")
def vgg16(weight_path):
    model = VGG(make_layers(cfgs['D'], batch_norm=False))
    model.eval()
    if weight_path:
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
    in_dict = {
        "model": model,
        "inputs": [torch.randn((1, 3, 224, 224))]
    }
    return in_dict

@onnx_infer_func.register('infer_imagenet_benchmark')
def infer_imagenet_benchmark(executor):
    return imagenet_benchmark(executor, crop_size=224)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def imagenet_benchmark(executor, crop_size=224):
    iters = executor.iteration
    batch_size = executor.batch_size
    print('iters:', iters)
    print('batch_size:', batch_size)
    valdir = executor.dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=executor.gpu >= 0,  # False for CPU
        sampler=None)

    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for i, (input, label) in enumerate(val_loader):
        input_numpy = input.numpy()
        output = executor.forward(input_numpy)
        output = torch.from_numpy(output[0]).data

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, label, topk=(1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            i, iters, batch_time=batch_time,
            top1=top1, top5=top5))

        if i + 1 == iters:
            break
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg.item()

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


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
@onnx_infer_func.register('infer_image')
def image(executor):
    image = executor.dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    img_rgb = Image.open(image).convert('RGB')
    img_tensor = image_preprocess(img_rgb)
    input_numpy = img_tensor.unsqueeze_(0).numpy()
    input_numpy *= 255
    input_numpy = input_numpy.astype(np.uint8)
    print(f'save model_input.bin to {executor.save_dir}')
    input_numpy.flatten().tofile(os.path.join(executor.save_dir, "model_input.bin"))
    output = executor.forward(input_numpy)
    write_numpy_to_file(output[0], os.path.join(executor.save_dir, "output.txt"))
    print(f'save output.txt to {executor.save_dir}')
    output = torch.from_numpy(output[0]).data
    # measure accuracy and record loss
    topk=(1, 5)
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    dicts = {}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'data/labels.txt')) as f:
        lines = f.readlines()
        for line in lines:
            num, label = line.strip('\n').split(':')
            dicts[num] = label
    index = pred.numpy().flatten()
    print('predict label top5:')
    for idx in index:
        print(idx, dicts[str(idx)])

    return
