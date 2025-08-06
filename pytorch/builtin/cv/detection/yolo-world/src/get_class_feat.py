# -*- coding: UTF-8 -*-
"""
@Project ：YOLO-World 
@IDE     ：PyCharm 
@Author  ：gxs
@Date    ：2025/2/18 星期二 14:22 
"""
import os
import itertools
from typing import List, Sequence, Tuple, Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import onnx
import onnxsim
import onnxruntime
from mmengine.config import Config
from mmdet.apis import init_detector
from mmdet.utils import OptMultiConfig, ConfigType

from transformers import AutoTokenizer, AutoModel, CLIPTextConfig
from transformers.models.clip.modeling_clip import (CLIPTextTransformer,
                                                    CLIPTextModelWithProjection,
                                                    CLIPTextModelOutput,
                                                    CLIP_TEXT_INPUTS_DOCSTRING)
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from yolo_world.models.backbones import HuggingCLIPLanguageBackbone


class CLIPTextTransformerNEW(CLIPTextTransformer):
    def __init__(self):
        pass

    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # # expand attention_mask
        # if attention_mask is not None and not self._use_flash_attention_2:
        #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #     attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        # (1, 77, 512)
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # (1, 512)
        # pooled_output = last_hidden_state[:, input_ids_index.to(device=last_hidden_state.device)]  #
        pooled_output = last_hidden_state  #

        # if self.eos_token_id == 2:
        #     # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
        #     # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
        #     # ------------------------------------------------------------
        #     # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        #     # take features from the eot embedding (eot_token is the highest number in each sequence)
        #     # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        #     pooled_output = last_hidden_state[
        #         torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        #             input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        #         # input_ids.to(device=last_hidden_state.device).argmax(dim=-1),
        #     ]
        # else:
        #     # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
        #     pooled_output = last_hidden_state[
        #         torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        #             # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
        #             # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
        #         (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
        #         .int()
        #         .argmax(dim=-1),
        #     ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPTextModelWithProjectionNEW(CLIPTextModelWithProjection):
    def __init__(self):
        pass

    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPTextModelOutput, config_class=CLIPTextConfig)
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPTextModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        # >>> from transformers import AutoTokenizer, CLIPTextModelWithProjection
        #
        # >>> model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        # >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        #
        # >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        #
        # >>> outputs = model(**inputs)
        # >>> text_embeds = outputs.text_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # (k, 77, 512)
        pooled_output = text_outputs[1]
        # (k, 77, 512)
        text_embeds = self.text_projection(pooled_output)

        if not return_dict:
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return CLIPTextModelOutput(
            text_embeds=text_embeds,  # (k, 77, 512)
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )


class HuggingCLIPLanguageBackboneNEW(HuggingCLIPLanguageBackbone):
    def __init__(self):
        pass

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        num_per_batch = input_ids.shape[0]

        txt_outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask)  # CLIPTextModelOutput
        txt_feats = txt_outputs.text_embeds  # Tensor: shape (b, 512)
        ## txt_feats.norm(p=2, dim=-1, keepdim=True) 计算特征的L2范数
        ## 把每个特征向量的长度缩放到1，即归一化为单位向量。
        # txt_feats_org = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)  # L2归一化

        # ### =========================================================================
        # # NOTE:因为Knight不支持直接调用的L2-norm算子，需要手动实现
        # # 计算步骤：平方，求和，开方
        # txt_l2_norm = torch.sqrt(torch.sum(torch.pow(txt_feats, 2), dim=-1, keepdim=True))
        # txt_feats = txt_feats / txt_l2_norm
        #
        # ### =========================================================================

        # txt_feats = txt_feats.reshape(-1, num_per_batch[0], txt_feats.shape[-1])
        # txt_feats = txt_feats.reshape(-1, num_per_batch, txt_feats.shape[-1])

        return txt_feats


if __name__ == '__main__':
    out_model_name = "text_encoder"
    config = "src/configs/finetune_coco/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py"
    checkpoint = "weight/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_ep80-492dc329.pth"
    names_path = "src/coco_names.txt"
    with open(names_path, "r", encoding="utf-8") as rFile:
        names = [line.strip() for line in rFile.readlines()]

    # texts = [["person"], ]
    texts = [[name, ] for name in names]
    cfg = Config.fromfile(config)

    ###=====================================================
    ### NOTE: 分词器 && 把输入文本转成向量
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_per_batch = [len(t) for t in texts]
    assert max(num_per_batch) == min(num_per_batch), ('number of sequences not equal in batch')
    texts = list(itertools.chain(*texts))
    # texts = tokenizer(text=texts, return_tensors='pt', padding="max_length")  # 最长77
    texts = tokenizer(text=texts, return_tensors='pt', padding=True)

    ### NOTE: 初始化模型
    model = init_detector(cfg, checkpoint=checkpoint, device="cpu", palette="coco")
    model.eval()

    ### NOTE: 文本编码器重写forward
    text_encoder_old = model.backbone.text_model
    # print(text_encoder_old, end="\n\n")
    text_encoder_new = HuggingCLIPLanguageBackboneNEW()
    text_encoder_new.__dict__.update(text_encoder_old.__dict__)

    ### NOTE: 只保留文本编码器
    text_encoder = text_encoder_new

    ### NOTE: 重新写它的forward && 替换源码中的模块
    text_model_new = CLIPTextModelWithProjectionNEW()
    text_model_old = text_encoder.model
    text_model_new.__dict__.update(text_model_old.__dict__)
    text_encoder.model = text_model_new

    ### NOTE: 重新写它的forward && 替换源码中的模块
    text_transform_new = CLIPTextTransformerNEW()
    text_transform_old = text_encoder.model.text_model
    text_transform_new.__dict__.update(text_transform_old.__dict__)
    text_encoder.model.text_model = text_transform_new

    ### ==========================================================
    ### NOTE: 获取输入
    input_ids: Tensor = texts.data["input_ids"]
    attention_mask: Tensor = texts.data["attention_mask"]

    #### NOTE: attention_mask前处理
    dtype = torch.float32
    bsz, src_len = attention_mask.size()
    tgt_len = src_len
    expanded_mask = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    attention_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    ### ==========================================================
    # print("input_ids: ")
    # print(input_ids)
    # 获取输出
    text_feats = text_encoder(input_ids, attention_mask)  # (k, 4, 512)

    ### NOTE: 后处理
    input_ids_index = input_ids.to(dtype=torch.int, ).argmax(dim=-1)
    text_feats = text_feats[torch.arange(text_feats.shape[0]), input_ids_index]  # (k, 512)
    print(text_feats.shape)

    text_l2_norm = torch.sqrt(torch.sum(torch.pow(text_feats, 2), dim=-1, keepdim=True))
    text_feats = text_feats / text_l2_norm
    text_feats = text_feats.reshape(-1, *text_feats.shape)
    text_feats = text_feats.detach().cpu().numpy()
    np.save(f"{names_path[:-4]}.npy",text_feats)

