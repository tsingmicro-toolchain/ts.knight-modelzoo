import math
import time
import os
import numpy as np
from PIL import Image
from copy import deepcopy
from functools import partial
from torchvision import datasets, transforms
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg, named_apply, checkpoint_seq
from timm.models.layers import PatchEmbed, Mlp, GluMlp, GatedMlp, DropPath, lecun_normal_, to_2tuple
from timm.models.registry import register_model
from onnx_quantize_tool.common.register import onnx_infer_func, pytorch_model

class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0), mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)


class ResBlock(nn.Module):
    """ Residual MLP block w/ LayerScale and Affine 'norm'

    Based on: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=4, mlp_layer=Mlp, norm_layer=Affine,
            act_layer=nn.GELU, init_values=1e-4, drop=0., drop_path=0.):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.linear_tokens = nn.Linear(seq_len, seq_len)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, drop=drop)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        return x


class SpatialGatingUnit(nn.Module):
    """ Spatial Gating Unit

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(self, dim, seq_len, norm_layer=nn.LayerNorm):
        super().__init__()
        gate_dim = dim // 2
        self.norm = norm_layer(gate_dim)
        self.proj = nn.Linear(seq_len, seq_len)

    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))
        return u * v.transpose(-1, -2)


class SpatialGatingBlock(nn.Module):
    """ Residual Block w/ Spatial Gating

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=4, mlp_layer=GatedMlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        sgu = partial(SpatialGatingUnit, seq_len=seq_len)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, gate_layer=sgu, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x

class MlpMixer(nn.Module):

    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            block_layer=MixerBlock,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
            global_pool='avg',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.grad_checkpointing = False

        self.stem = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=norm_layer if stem_norm else None)
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim, self.stem.num_patches, mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer,
                act_layer=act_layer, drop=drop_rate, drop_path=drop_path_rate)
            for _ in range(num_blocks)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(nlhb=nlhb)

    @torch.jit.ignore
    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head(x)
        return x


def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()


def checkpoint_filter_fn(state_dict, model):
    """ Remap checkpoints if needed """
    if 'patch_embed.proj.weight' in state_dict:
        # Remap FB ResMlp models -> timm
        out_dict = {}
        for k, v in state_dict.items():
            k = k.replace('patch_embed.', 'stem.')
            k = k.replace('attn.', 'linear_tokens.')
            k = k.replace('mlp.', 'mlp_channels.')
            k = k.replace('gamma_', 'ls')
            if k.endswith('.alpha') or k.endswith('.beta'):
                v = v.reshape(1, 1, -1)
            out_dict[k] = v
        return out_dict
    return state_dict

def _create_mixer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for MLP-Mixer models.')

    model = build_model_with_cfg(
        MlpMixer, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model

@pytorch_model.register("mixer_b16_224")
def mixer_b16_224(weight_path=None):
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768)
    model = _create_mixer('mixer_b16_224', pretrained=False, **model_args)
    model.eval()

    if weight_path:
        state_dict = torch.load(weight_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)

    return {"model": model,  "inputs": [torch.randn(1,3,224,224)]}


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
            #normalize,
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
        input_numpy = input.numpy()*255
        input_numpy = input_numpy.astype(np.uint8)
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
        #normalize,
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