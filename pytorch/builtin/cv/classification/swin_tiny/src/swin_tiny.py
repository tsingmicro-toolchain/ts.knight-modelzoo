import torch
import torchvision
from Swin_Transformer.models.swin_transformer import SwinTransformer
from Swin_Transformer.models.swin_transformer_v2 import SwinTransformerV2
from timm.utils import accuracy, AverageMeter
from torchvision import datasets, transforms
from onnx_quantize_tool.common.register import onnx_infer_func, pytorch_model


@pytorch_model.register("swin_tiny")
def swin_tiny(weight_path=None):
    model = SwinTransformer(img_size=224,
                            patch_size=4,
                            in_chans=3,
                            num_classes=1000,
                            embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            drop_rate=0.0,
                            drop_path_rate=0.2,
                            ape=False,
                            patch_norm=True,
                            use_checkpoint=False)
    if weight_path:
        weight_state_dict = torch.load(weight_path, map_location=torch.device('cpu'))["model"]
        model.load_state_dict(weight_state_dict, strict=True)
    return {"model": model,  "inputs": [torch.randn(10,3,224,224)]}


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


@onnx_infer_func.register('infer_imagenet_benchmark')
def infer_imagenet_benchmark(executor):
    return imagenet_benchmark(executor, crop_size=224)