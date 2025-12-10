import copy
import math
import warnings
import einops
import mmengine
import numpy as np
import torch
from mmcv.cnn import build_norm_layer, ConvModule
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.ops import point_sample
from mmengine import ConfigDict, to_2tuple
from mmengine.dist import is_main_process
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmengine.structures import InstanceData
from peft import get_peft_config, get_peft_model
from torch import nn, Tensor
from transformers import SamConfig
from transformers.models.sam.modeling_sam import SamPatchEmbeddings,SamVisionNeck,SamVisionAttention,SamMaskDecoder, SamPositionalEmbedding, \
    SamPromptEncoder, SamModel, SamVisionEncoderOutput
from typing import List, Tuple, Optional, Dict, Union, Sequence
from mmdet.models import MaskRCNN, StandardRoIHead, FCNMaskHead, SinePositionalEncoding, Mask2Former, Mask2FormerHead, \
    MaskFormerFusionHead, BaseDetector
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import unpack_gt_instances, empty_instances, multi_apply, \
    get_uncertain_point_coords_with_randomness
from mmdet.registry import MODELS
from mmdet.structures import SampleList, DetDataSample, OptSampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import OptConfigType, MultiConfig, ConfigType, InstanceList, reduce_mean
import torch.nn.functional as F

from mmpretrain.models import LayerNorm2d
from mmpretrain.models.utils import build_norm_layer as build_norm_layer_pretrain
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from mmengine.runner.checkpoint import load_checkpoint
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from transformers import SamConfig
from peft import get_peft_model, get_peft_config
class SamVisionEncoder(nn.Module):
    def __init__(self, config, num_experts=8, top_k=2, noise_scale=0.1):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = SamPatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = SamVisionLayer(
                config,
                window_size=0,#config.window_size if i not in config.global_attn_indexes else 0,
                #num_experts=num_experts,
                #top_k=top_k,
                #noise_scale=noise_scale
            )
            self.layers.append(layer)

        self.neck = SamVisionNeck(config)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.patch_embed

    def forward(
        self,
        pixel_values: Optional[torch.cuda.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SamVisionEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        #print("after patch embedding=",hidden_states.shape)#################################################
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed
            #print("if self.pos_embed is not None=", hidden_states.shape)
            #print("self.pos_embed=", self.pos_embed.shape)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                #print("after ",i," layer,if output_hidden_states:", all_hidden_states[0].shape)
                #reshaped_x = all_hidden_states[0].view(1, 64 * 64, 768)
                #print("????????????????????:",reshaped_x.shape)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]
            #print("after i layer,if output_hidden_states:", hidden_states.shape)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.neck(hidden_states)
        #print("after self.neck ",all_hidden_states[2].shape)#################################################
        #print("after self.neck hidden states", hidden_states.shape)
        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class NoisyMixtureOfExperts(nn.Module):
    def __init__(self, dim, num_experts, expert_dim, top_k=1, noise_scale=0.1):
        super().__init__()
        self.dim=dim#768
        self.expert_dim=expert_dim#8
        self.num_experts = 1536#num_experts#3072
        self.top_k = 1#top_k
        self.noise_scale = noise_scale
        self.experts = nn.ModuleList([nn.Linear(dim, expert_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)#768,3072
        self.projection = nn.Linear(expert_dim, dim)

    def forward(self, x):
        """
        输入 x 的形状为 (1, 64, 64, 768)
        输出形状为 (1, 64, 64, 768)
        """
        # 保存原始形状
        #print("dim",self.dim)
        #print("num_experts",self.num_experts)
        #print("expert_dim",self.expert_dim)
        original_shape = x.shape  # (1, 64, 64, 768)
        batch_size, height, width, dim = original_shape
        #print("original_shape",original_shape)
        # 将输入展平为 (batch_size * height * width, dim)
        x_flat = x.reshape(-1, dim)  # (4096, 768)

        # 计算门控权重
        gate_scores = self.gate(x_flat)  # (4096, 3072)
        noise = torch.randn_like(gate_scores) * self.noise_scale  # 添加噪声(4096, 3072)
        gate_scores = gate_scores + noise#(4096, 3072)
        gate_weights = F.softmax(gate_scores, dim=-1)  # 归一化 (4096, 3072)

        # 选择 Top-K 专家
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # (4096, top_k)
        #print("top_k_weights",top_k_weights.shape)# (4096,2)
        #print("top_k_indices",top_k_indices.shape)# (4096,2)
        # 计算专家输出
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=-1)  # (4096, 3072, 8)
        #print("expert_outputs",expert_outputs.shape)
        # 加权组合 Top-K 专家输出
        output_flat = torch.zeros_like(x_flat)  # 初始化输出，形状为 (4096, dim)
        #print("output_flat",output_flat.shape)
        #print("top_k",self.top_k)
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]  # 当前选择的专家索引，形状为 (4096,1)
            expert_weight = top_k_weights[..., i]  # 当前专家的权重，形状为 (4096,1)

            # 选择专家输出
            expert_output = expert_outputs[torch.arange(expert_outputs.size(0)), :, expert_idx]  # (4096, 3072)
            #print("expert_output",expert_output.shape)
            # 将专家输出投影回原始维度
            expert_output = self.projection(expert_output)  # (4096, 768)
            #print("expert_output",expert_output.shape)
            # 加权累加
            output_flat += expert_output * expert_weight.unsqueeze(-1)  # (4096, 768)

        # 将输出恢复为原始形状
        output = output_flat.reshape(original_shape)  # (1, 64, 64, 768)
        return output

class NoisyMixtureOfExperts1(nn.Module):
    def __init__(self, dim, num_experts, expert_dim, top_k=1, noise_scale=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_scale = noise_scale
        self.experts = nn.ModuleList([nn.Linear(dim, expert_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        """
        输入 x 的形状为 (1, 64, 64, 768)
        输出形状为 (1, 64, 64, 768)
        """
        # 保存原始形状
        original_shape = x.shape  # (1, 64, 64, 768)
        batch_size, height, width, dim = original_shape

        # 将输入展平为 (batch_size * height * width, dim)
        x_flat = x.reshape(-1, dim)  # (4096, 768)

        # 计算门控权重
        gate_scores = self.gate(x_flat)  # (4096, num_experts)
        noise = torch.randn_like(gate_scores) * self.noise_scale  # 添加噪声
        gate_scores = gate_scores + noise
        gate_weights = F.softmax(gate_scores, dim=-1)  # 归一化

        # 选择 Top-K 专家
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # (4096, top_k)

        # 计算专家输出
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=-1)  # (4096, expert_dim, num_experts)

        # 加权组合 Top-K 专家输出
        output_flat = torch.zeros_like(x_flat)  # 初始化输出，形状为 (4096, expert_dim)
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]  # 当前选择的专家索引，形状为 (4096,)
            expert_weight = top_k_weights[..., i]  # 当前专家的权重，形状为 (4096,)
            print("expert_weight",expert_weight.shape)
            # 选择专家输出
            expert_output = expert_outputs[torch.arange(expert_outputs.size(0)), :, expert_idx]  # (4096, expert_dim)
            print("expert_output",expert_output.shape)
            # 加权累加
            output_flat += expert_output * expert_weight.unsqueeze(-1)  # (4096, expert_dim)

        # 将输出恢复为原始形状
        output = output_flat.reshape(original_shape)  # (1, 64, 64, 768)
        return output

class NoisyMixtureOfExperts_seek(nn.Module):
    def __init__(self, dim, num_experts, expert_dim, top_k=1, noise_scale=0.1):
        super().__init__()
        self.dim=dim#768
        self.expert_dim=3072#8
        self.num_experts =8#num_experts#3072
        self.top_k = 1#top_k
        self.noise_scale = noise_scale
        self.experts = nn.ModuleList([nn.Linear(dim, self.expert_dim) for _ in range(self.num_experts)])
        self.gate = nn.Linear(dim, self.num_experts)#768,3072
        self.projection = nn.Linear(self.expert_dim, dim)

    def forward(self, x):
        """
        输入 x 的形状为 (1, 64, 64, 768)
        输出形状为 (1, 64, 64, 768)
        """
        # 保存原始形状
        #print("dim",self.dim)
        #print("num_experts",self.num_experts)
        #print("expert_dim",self.expert_dim)
        original_shape = x.shape  # (1, 64, 64, 768)
        batch_size, height, width, dim = original_shape
        #print("original_shape",original_shape)
        # 将输入展平为 (batch_size * height * width, dim)
        x_flat = x.reshape(-1, dim)  # (4096, 768)

        # 计算门控权重
        gate_scores = self.gate(x_flat)  # (4096, 3072)
        noise = torch.randn_like(gate_scores) * self.noise_scale  # 添加噪声(4096, 3072)
        gate_scores = gate_scores + noise#(4096, 3072)
        gate_weights = F.softmax(gate_scores, dim=-1)  # 归一化 (4096, 3072)

        # 选择 Top-K 专家
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # (4096, top_k)
        #print("top_k_weights",top_k_weights.shape)# (4096,2)
        #print("top_k_indices",top_k_indices.shape)# (4096,2)
        # 计算专家输出
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=-1)  # (4096, 3072, 8)
        #print("expert_outputs",expert_outputs.shape)
        # 加权组合 Top-K 专家输出
        output_flat = torch.zeros_like(x_flat)  # 初始化输出，形状为 (4096, dim)
        #print("output_flat",output_flat.shape)
        #print("top_k",self.top_k)
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]  # 当前选择的专家索引，形状为 (4096,1)
            expert_weight = top_k_weights[..., i]  # 当前专家的权重，形状为 (4096,1)

            # 选择专家输出
            expert_output = expert_outputs[torch.arange(expert_outputs.size(0)), :, expert_idx]  # (4096, 3072)
            #print("expert_output",expert_output.shape)
            # 将专家输出投影回原始维度
            expert_output = self.projection(expert_output)  # (4096, 768)
            #print("expert_output",expert_output.shape)
            # 加权累加
            output_flat += expert_output * expert_weight.unsqueeze(-1)  # (4096, 768)

        # 将输出恢复为原始形状
        output = output_flat.reshape(original_shape)  # (1, 64, 64, 768)
        return output

class SamVisionLayer(nn.Module):
    def __init__(self, config, window_size, num_experts=8, top_k=2, noise_scale=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = SamVisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 将 MLP 替换为 MOE
        self.moe = NoisyMixtureOfExperts_seek(
            dim=config.hidden_size,
            num_experts=num_experts,  # 专家数量
            expert_dim=config.mlp_dim,  # 专家维度
            top_k=top_k,  # 每次激活的专家数量
            noise_scale=noise_scale  # 噪声比例
        )
        self.window_size = window_size

    def window_partition(self, hidden_states: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        将输入张量划分为非重叠的窗口，并在需要时进行填充。
        Args:
            hidden_states: 输入张量，形状为 [batch_size, height, width, channel]。
            window_size: 窗口大小。
        Returns:
            windows: 划分后的窗口，形状为 [batch_size * num_windows, window_size, window_size, channel]。
            (pad_height, pad_width): 划分前的填充高度和宽度。
        """
        batch_size, height, width, channel = hidden_states.shape

        # 计算需要填充的高度和宽度
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        pad_height, pad_width = height + pad_h, width + pad_w

        # 划分窗口
        hidden_states = hidden_states.reshape(
            batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel
        )
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, channel)
        return windows, (pad_height, pad_width)

    def window_unpartition(
        self, windows: torch.Tensor, window_size: int, padding_shape: Tuple[int, int], original_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        将窗口反划分为原始序列，并移除填充。
        Args:
            windows: 输入窗口，形状为 [batch_size * num_windows, window_size, window_size, channel]。
            window_size: 窗口大小。
            padding_shape: 填充后的高度和宽度 (pad_height, pad_width)。
            original_shape: 原始高度和宽度 (height, width)。
        Returns:
            hidden_states: 反划分后的序列，形状为 [batch_size, height, width, channel]。
        """
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = windows.shape[0] // (pad_height * pad_width // window_size // window_size)

        # 反划分窗口
        hidden_states = windows.reshape(
            batch_size, pad_height // window_size, pad_width // window_size, window_size, window_size, -1
        )
        hidden_states = (
            hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch_size, pad_height, pad_width, -1)
        )

        # 移除填充
        hidden_states = hidden_states[:, :height, :width, :].contiguous()
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # Window partition
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)

        hidden_states, attn_weights = self.attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        # Reverse window partition
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))

        hidden_states = residual + hidden_states
        layernorm_output = self.layer_norm2(hidden_states)
        #print("hidden_states",hidden_states.shape)
        #print("layernorm_output", layernorm_output.shape)
        #print("self.moe(layernorm_output)",self.moe(layernorm_output).shape)
        hidden_states = hidden_states + self.moe(layernorm_output)  # 使用 MOE 替换 MLP

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

@MODELS.register_module()
class RSSamVisionEncoder_moe(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            peft_config=None,
            init_cfg=None,
            prune_amount=0.2,  # 剪枝比例
            num_experts=8,  # 专家数量
            top_k=2,  # 每次激活的专家数量
            noise_scale=0.1  # 噪声比例
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        vision_encoder = SamVisionEncoder(sam_config, num_experts, top_k, noise_scale)  # 修改为支持 MOE

        # 加载预训练权重
        if init_cfg is not None:
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[(r'^module\.', ''), (r'^vision_encoder\.', '')])

        # 初始化掩码字典
        self.masks = {}  # 用于保存每个模块的掩码

        # 对模型进行剪枝
        self.prune_amount = prune_amount
        self.prune_model(vision_encoder, amount=prune_amount)

        # 配置 PEFT（如果需要）
        if peft_config is not None and isinstance(peft_config, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_config)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder

        self.vision_encoder.is_init = True

    def prune_model(self, model, amount=0.2):
        """
        对模型的权重进行剪枝，并保存掩码
        :param model: 需要剪枝的模型
        :param amount: 剪枝比例
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):  # 对全连接层进行剪枝
                # 使用 L1 范数剪枝
                prune.l1_unstructured(module, name='weight', amount=amount)
                # 保存掩码
                self.masks[name] = module.weight_mask.clone()
                # 永久移除剪枝的权重
                prune.remove(module, 'weight')

    def apply_mask(self):
        """
        在训练过程中应用掩码，固定被剪枝的权重为 0
        """
        for name, module in self.vision_encoder.named_modules():
            if isinstance(module, nn.Linear):
                #print("1111111111111111")
                if name in self.masks:  # 检查是否有掩码
                    #print("222222222222222222222")
                    #print(self.masks)
                    #print(name)
                    # 将掩码移动到与模型权重相同的设备上
                    mask = self.masks[name].to(module.weight.device)
                    # 应用掩码，固定被剪枝的权重为 0
                    module.weight.data.mul_(mask)

    def forward(self, *args, **kwargs):
        # 在每次前向传播时应用掩码
        self.apply_mask()
        return self.vision_encoder(*args, **kwargs)

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')


