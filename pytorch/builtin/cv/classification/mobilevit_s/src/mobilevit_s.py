import math
import time
import os
import numpy as np
from PIL import Image
from typing import Union, Callable, Dict, Tuple, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from timm.models.byobnet import register_block, ByoBlockCfg, ByoModelCfg, ByobNet, LayerFn, num_groups
from timm.models.fx_features import register_notrace_module
from timm.models.layers import to_2tuple, make_divisible, LayerNorm2d, GroupNorm1, ConvMlp, DropPath
from timm.models.vision_transformer import Block as TransformerBlock
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model
from onnx_quantize_tool.common.register import onnx_infer_func, pytorch_model

__all__ = []


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 256, 256), 'pool_size': (8, 8),
        'crop_pct': 0.9, 'interpolation': 'bicubic',
        'mean': (0., 0., 0.), 'std': (1., 1., 1.),
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        'fixed_input_size': False,
        **kwargs
    }

def _inverted_residual_block(d, c, s, br=4.0):
    # inverted residual is a bottleneck block with bottle_ratio > 1 applied to in_chs, linear output, gs=1 (depthwise)
    return ByoBlockCfg(
        type='bottle', d=d, c=c, s=s, gs=1, br=br,
        block_kwargs=dict(bottle_in=True, linear_out=True))


def _mobilevit_block(d, c, s, transformer_dim, transformer_depth, patch_size=4, br=4.0):
    # inverted residual + mobilevit blocks as per MobileViT network
    return (
        _inverted_residual_block(d=d, c=c, s=s, br=br),
        ByoBlockCfg(
            type='mobilevit', d=1, c=c, s=1,
            block_kwargs=dict(
                transformer_dim=transformer_dim,
                transformer_depth=transformer_depth,
                patch_size=patch_size)
        )
    )


def _mobilevitv2_block(d, c, s, transformer_depth, patch_size=2, br=2.0, transformer_br=0.5):
    # inverted residual + mobilevit blocks as per MobileViT network
    return (
        _inverted_residual_block(d=d, c=c, s=s, br=br),
        ByoBlockCfg(
            type='mobilevit2', d=1, c=c, s=1, br=transformer_br, gs=1,
            block_kwargs=dict(
                transformer_depth=transformer_depth,
                patch_size=patch_size)
        )
    )


def _mobilevitv2_cfg(multiplier=1.0):
    chs = (64, 128, 256, 384, 512)
    if multiplier != 1.0:
        chs = tuple([int(c * multiplier) for c in chs])
    cfg = ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=chs[0], s=1, br=2.0),
            _inverted_residual_block(d=2, c=chs[1], s=2, br=2.0),
            _mobilevitv2_block(d=1, c=chs[2], s=2, transformer_depth=2),
            _mobilevitv2_block(d=1, c=chs[3], s=2, transformer_depth=4),
            _mobilevitv2_block(d=1, c=chs[4], s=2, transformer_depth=3),
        ),
        stem_chs=int(32 * multiplier),
        stem_type='3x3',
        stem_pool='',
        downsample='',
        act_layer='silu',
    )
    return cfg


model_cfgs = dict(
    mobilevit_xxs=ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=16, s=1, br=2.0),
            _inverted_residual_block(d=3, c=24, s=2, br=2.0),
            _mobilevit_block(d=1, c=48, s=2, transformer_dim=64, transformer_depth=2, patch_size=2, br=2.0),
            _mobilevit_block(d=1, c=64, s=2, transformer_dim=80, transformer_depth=4, patch_size=2, br=2.0),
            _mobilevit_block(d=1, c=80, s=2, transformer_dim=96, transformer_depth=3, patch_size=2, br=2.0),
        ),
        stem_chs=16,
        stem_type='3x3',
        stem_pool='',
        downsample='',
        act_layer='silu',
        num_features=320,
    ),

    mobilevit_xs=ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=32, s=1),
            _inverted_residual_block(d=3, c=48, s=2),
            _mobilevit_block(d=1, c=64, s=2, transformer_dim=96, transformer_depth=2, patch_size=2),
            _mobilevit_block(d=1, c=80, s=2, transformer_dim=120, transformer_depth=4, patch_size=2),
            _mobilevit_block(d=1, c=96, s=2, transformer_dim=144, transformer_depth=3, patch_size=2),
        ),
        stem_chs=16,
        stem_type='3x3',
        stem_pool='',
        downsample='',
        act_layer='silu',
        num_features=384,
    ),

    mobilevit_s=ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=32, s=1),
            _inverted_residual_block(d=3, c=64, s=2),
            _mobilevit_block(d=1, c=96, s=2, transformer_dim=144, transformer_depth=2, patch_size=2),
            _mobilevit_block(d=1, c=128, s=2, transformer_dim=192, transformer_depth=4, patch_size=2),
            _mobilevit_block(d=1, c=160, s=2, transformer_dim=240, transformer_depth=3, patch_size=2),
        ),
        stem_chs=16,
        stem_type='3x3',
        stem_pool='',
        downsample='',
        act_layer='silu',
        num_features=640,
    ),

    semobilevit_s=ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=32, s=1),
            _inverted_residual_block(d=3, c=64, s=2),
            _mobilevit_block(d=1, c=96, s=2, transformer_dim=144, transformer_depth=2, patch_size=2),
            _mobilevit_block(d=1, c=128, s=2, transformer_dim=192, transformer_depth=4, patch_size=2),
            _mobilevit_block(d=1, c=160, s=2, transformer_dim=240, transformer_depth=3, patch_size=2),
        ),
        stem_chs=16,
        stem_type='3x3',
        stem_pool='',
        downsample='',
        attn_layer='se',
        attn_kwargs=dict(rd_ratio=1/8),
        num_features=640,
    ),

    mobilevitv2_050=_mobilevitv2_cfg(.50),
    mobilevitv2_075=_mobilevitv2_cfg(.75),
    mobilevitv2_125=_mobilevitv2_cfg(1.25),
    mobilevitv2_100=_mobilevitv2_cfg(1.0),
    mobilevitv2_150=_mobilevitv2_cfg(1.5),
    mobilevitv2_175=_mobilevitv2_cfg(1.75),
    mobilevitv2_200=_mobilevitv2_cfg(2.0),
)


@register_notrace_module
class MobileVitBlock(nn.Module):
    """ MobileViT block
        Paper: https://arxiv.org/abs/2110.02178?context=cs.LG
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            kernel_size: int = 3,
            stride: int = 1,
            bottle_ratio: float = 1.0,
            group_size: Optional[int] = None,
            dilation: Tuple[int, int] = (1, 1),
            mlp_ratio: float = 2.0,
            transformer_dim: Optional[int] = None,
            transformer_depth: int = 2,
            patch_size: int = 8,
            num_heads: int = 4,
            attn_drop: float = 0.,
            drop: int = 0.,
            no_fusion: bool = False,
            drop_path_rate: float = 0.,
            layers: LayerFn = None,
            transformer_norm_layer: Callable = nn.LayerNorm,
            **kwargs,  # eat unused args
    ):
        super(MobileVitBlock, self).__init__()

        layers = layers or LayerFn()
        groups = num_groups(group_size, in_chs)
        out_chs = out_chs or in_chs
        transformer_dim = transformer_dim or make_divisible(bottle_ratio * in_chs)

        self.conv_kxk = layers.conv_norm_act(
            in_chs, in_chs, kernel_size=kernel_size,
            stride=stride, groups=groups, dilation=dilation[0])
        self.conv_1x1 = nn.Conv2d(in_chs, transformer_dim, kernel_size=1, bias=False)

        self.transformer = nn.Sequential(*[
            TransformerBlock(
                transformer_dim, mlp_ratio=mlp_ratio, num_heads=num_heads, qkv_bias=True,
                attn_drop=attn_drop, drop_path=drop_path_rate,
                act_layer=layers.act, norm_layer=transformer_norm_layer)
            for _ in range(transformer_depth)
        ])
        self.norm = transformer_norm_layer(transformer_dim)

        self.conv_proj = layers.conv_norm_act(transformer_dim, out_chs, kernel_size=1, stride=1)

        if no_fusion:
            self.conv_fusion = None
        else:
            self.conv_fusion = layers.conv_norm_act(in_chs + out_chs, out_chs, kernel_size=kernel_size, stride=1)

        self.patch_size = to_2tuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # Local representation
        x = self.conv_kxk(x)
        x = self.conv_1x1(x)

        # Unfold (feature map -> patches)
        patch_h, patch_w = self.patch_size
        B, C, H, W = x.shape
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w
        num_patches = num_patch_h * num_patch_w  # N
        interpolate = False
        if new_h != H or new_w != W:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # [B, C, H, W] --> [B * C * n_h, n_w, p_h, p_w]
        x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [BP, N, C] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)

        # Global representations
        x = self.transformer(x)
        x = self.norm(x)

        # Fold (patch -> feature map)
        # [B, P, N, C] --> [B*C*n_h, n_w, p_h, p_w]
        x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        if interpolate:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        x = self.conv_proj(x)
        if self.conv_fusion is not None:
            x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        return x


class LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in `https://arxiv.org/abs/2206.02680`
    This layer can be used for self- as well as cross-attention.
    Args:
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_drop (float): Dropout value for context scores. Default: 0.0
        bias (bool): Use bias in learnable layers. Default: True
    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input
    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
        self,
        embed_dim: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.qkv_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
        )
        self.out_drop = nn.Dropout(proj_drop)

    def _forward_self_attn(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = qkv.split([1, self.embed_dim, self.embed_dim], dim=1)

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N] --> [B, d, P, 1]
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out

    @torch.jit.ignore()
    def _forward_cross_attn(self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x --> [B, C, P, N]
        # x_prev = [B, C, P, M]
        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape
        q_patch_area, q_num_patches = x.shape[-2:]

        assert (
            kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"

        # compute query, key, and value
        # [B, C, P, M] --> [B, 1 + d, P, M]
        qk = F.conv2d(
            x_prev,
            weight=self.qkv_proj.weight[:self.embed_dim + 1],
            bias=self.qkv_proj.bias[:self.embed_dim + 1],
        )

        # [B, 1 + d, P, M] --> [B, 1, P, M], [B, d, P, M]
        query, key = qk.split([1, self.embed_dim], dim=1)
        # [B, C, P, N] --> [B, d, P, N]
        value = F.conv2d(
            x,
            weight=self.qkv_proj.weight[self.embed_dim + 1],
            bias=self.qkv_proj.bias[self.embed_dim + 1] if self.qkv_proj.bias is not None else None,
        )

        # apply softmax along M dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)

        # compute context vector
        # [B, d, P, M] * [B, 1, P, M] -> [B, d, P, M] --> [B, d, P, 1]
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out

    def forward(self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_prev is None:
            return self._forward_self_attn(x)
        else:
            return self._forward_cross_attn(x, x_prev=x_prev)


class LinearTransformerBlock(nn.Module):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 paper <>`_
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        mlp_ratio (float): Inner dimension ratio of the FFN relative to embed_dim
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Dropout rate for attention in multi-head attention. Default: 0.0
        drop_path (float): Stochastic depth rate Default: 0.0
        norm_layer (Callable): Normalization layer. Default: layer_norm_2d
    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        act_layer = act_layer or nn.SiLU
        norm_layer = norm_layer or GroupNorm1

        self.norm1 = norm_layer(embed_dim)
        self.attn = LinearSelfAttention(embed_dim=embed_dim, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = norm_layer(embed_dim)
        self.mlp = ConvMlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_prev is None:
            # self-attention
            x = x + self.drop_path1(self.attn(self.norm1(x)))
        else:
            # cross-attention
            res = x
            x = self.norm1(x)  # norm
            x = self.attn(x, x_prev)  # attn
            x = self.drop_path1(x) + res  # residual

        # Feed forward network
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


@register_notrace_module
class MobileVitV2Block(nn.Module):
    """
    This class defines the `MobileViTv2 block <>`_
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: Optional[int] = None,
        kernel_size: int = 3,
        bottle_ratio: float = 1.0,
        group_size: Optional[int] = 1,
        dilation: Tuple[int, int] = (1, 1),
        mlp_ratio: float = 2.0,
        transformer_dim: Optional[int] = None,
        transformer_depth: int = 2,
        patch_size: int = 8,
        attn_drop: float = 0.,
        drop: int = 0.,
        drop_path_rate: float = 0.,
        layers: LayerFn = None,
        transformer_norm_layer: Callable = GroupNorm1,
        **kwargs,  # eat unused args
    ):
        super(MobileVitV2Block, self).__init__()
        layers = layers or LayerFn()
        groups = num_groups(group_size, in_chs)
        out_chs = out_chs or in_chs
        transformer_dim = transformer_dim or make_divisible(bottle_ratio * in_chs)

        self.conv_kxk = layers.conv_norm_act(
            in_chs, in_chs, kernel_size=kernel_size,
            stride=1, groups=groups, dilation=dilation[0])
        self.conv_1x1 = nn.Conv2d(in_chs, transformer_dim, kernel_size=1, bias=False)

        self.transformer = nn.Sequential(*[
            LinearTransformerBlock(
                transformer_dim,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path_rate,
                act_layer=layers.act,
                norm_layer=transformer_norm_layer
            )
            for _ in range(transformer_depth)
        ])
        self.norm = transformer_norm_layer(transformer_dim)

        self.conv_proj = layers.conv_norm_act(transformer_dim, out_chs, kernel_size=1, stride=1, apply_act=False)

        self.patch_size = to_2tuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        patch_h, patch_w = self.patch_size
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w
        num_patches = num_patch_h * num_patch_w  # N
        if new_h != H or new_w != W:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)

        # Local representation
        x = self.conv_kxk(x)
        x = self.conv_1x1(x)

        # Unfold (feature map -> patches), [B, C, H, W] -> [B, C, P, N]
        C = x.shape[1]
        x = x.reshape(B, C, num_patch_h, patch_h, num_patch_w, patch_w).permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C, -1, num_patches)

        # Global representations
        x = self.transformer(x)
        x = self.norm(x)

        # Fold (patches -> feature map), [B, C, P, N] --> [B, C, H, W]
        x = x.reshape(B, C, patch_h, patch_w, num_patch_h, num_patch_w).permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)

        x = self.conv_proj(x)
        return x


register_block('mobilevit', MobileVitBlock)
register_block('mobilevit2', MobileVitV2Block)


def _create_mobilevit(variant, cfg_variant=None, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ByobNet, variant, pretrained,
        model_cfg=model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        **kwargs)


def _create_mobilevit2(variant, cfg_variant=None, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ByobNet, variant, pretrained,
        model_cfg=model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        **kwargs)

@pytorch_model.register("mobilevit_s")
def mobilevit_s(weight_path=None):
    model = _create_mobilevit('mobilevit_s')
    model.eval()
    if weight_path:
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
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