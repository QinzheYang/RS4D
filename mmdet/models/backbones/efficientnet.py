# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import ConvModule, DropPath
from mmengine.model import BaseModule, Sequential

from mmdet.registry import MODELS
from ..layers import InvertedResidual, SELayer
from ..utils import make_divisible


class EdgeResidual(BaseModule):
    """Edge Residual Block.

    Args:
        in_channels (int): The input channels of this module.
        out_channels (int): The output channels of this module.
        mid_channels (int): The input channels of the second convolution.
        kernel_size (int): The kernel size of the first convolution.
            Defaults to 3.
        stride (int): The stride of the first convolution. Defaults to 1.
        se_cfg (dict, optional): Config dict for se layer. Defaults to None,
            which means no se layer.
        with_residual (bool): Use residual connection. Defaults to True.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict | list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 kernel_size=3,
                 stride=1,
                 se_cfg=None,
                 with_residual=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 drop_path_rate=0.,
                 with_cp=False,
                 init_cfg=None,
                 **kwargs):
        super(EdgeResidual, self).__init__(init_cfg=init_cfg)
        assert stride in [1, 2]
        self.with_cp = with_cp
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.with_se = se_cfg is not None
        self.with_residual = (
            stride == 1 and in_channels == out_channels and with_residual)

        if self.with_se:
            assert isinstance(se_cfg, dict)

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if self.with_se:
            self.se = SELayer(**se_cfg)

        self.conv2 = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):

        def _inner_forward(x):
            out = x
            out = self.conv1(out)

            if self.with_se:
                out = self.se(out)

            out = self.conv2(out)

            if self.with_residual:
                return x + self.drop_path(out)
            else:
                return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


def model_scaling(layer_setting, arch_setting):
    """Scaling operation to the layer's parameters according to the
    arch_setting."""
    # scale width
    new_layer_setting = copy.deepcopy(layer_setting)
    for layer_cfg in new_layer_setting:
        for block_cfg in layer_cfg:
            block_cfg[1] = make_divisible(block_cfg[1] * arch_setting[0], 8)

    # scale depth
    split_layer_setting = [new_layer_setting[0]]
    for layer_cfg in new_layer_setting[1:-1]:
        tmp_index = [0]
        for i in range(len(layer_cfg) - 1):
            if layer_cfg[i + 1][1] != layer_cfg[i][1]:
                tmp_index.append(i + 1)
        tmp_index.append(len(layer_cfg))
        for i in range(len(tmp_index) - 1):
            split_layer_setting.append(layer_cfg[tmp_index[i]:tmp_index[i +
                                                                        1]])
    split_layer_setting.append(new_layer_setting[-1])

    num_of_layers = [len(layer_cfg) for layer_cfg in split_layer_setting[1:-1]]
    new_layers = [
        int(math.ceil(arch_setting[1] * num)) for num in num_of_layers
    ]

    merge_layer_setting = [split_layer_setting[0]]
    for i, layer_cfg in enumerate(split_layer_setting[1:-1]):
        if new_layers[i] <= num_of_layers[i]:
            tmp_layer_cfg = layer_cfg[:new_layers[i]]
        else:
            tmp_layer_cfg = copy.deepcopy(layer_cfg) + [layer_cfg[-1]] * (
                new_layers[i] - num_of_layers[i])
        if tmp_layer_cfg[0][3] == 1 and i != 0:
            merge_layer_setting[-1] += tmp_layer_cfg.copy()
        else:
            merge_layer_setting.append(tmp_layer_cfg.copy())
    merge_layer_setting.append(split_layer_setting[-1])

    return merge_layer_setting


@MODELS.register_module()
class EfficientNet(BaseModule):
    """EfficientNet backbone.

    Args:
        arch (str): Architecture of efficientnet. Defaults to b0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (6, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    """

    # Parameters to build layers.
    # 'b' represents the architecture of normal EfficientNet family includes
    # 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'.
    # 'e' represents the architecture of EfficientNet-EdgeTPU including 'es',
    # 'em', 'el'.
    # 6 parameters are needed to construct a layer, From left to right:
    # - kernel_size: The kernel size of the block
    # - out_channel: The number of out_channels of the block
    # - se_ratio: The sequeeze ratio of SELayer.
    # - stride: The stride of the block
    # - expand_ratio: The expand_ratio of the mid_channels
    # - block_type: -1: Not a block, 0: InvertedResidual, 1: EdgeResidual
    layer_settings = {
        'b': [[[3, 32, 0, 2, 0, -1]],
              [[3, 16, 4, 1, 1, 0]],
              [[3, 24, 4, 2, 6, 0],
               [3, 24, 4, 1, 6, 0]],
              [[5, 40, 4, 2, 6, 0],
               [5, 40, 4, 1, 6, 0]],
              [[3, 80, 4, 2, 6, 0],
               [3, 80, 4, 1, 6, 0],
               [3, 80, 4, 1, 6, 0],
               [5, 112, 4, 1, 6, 0],
               [5, 112, 4, 1, 6, 0],
               [5, 112, 4, 1, 6, 0]],
              [[5, 192, 4, 2, 6, 0],
               [5, 192, 4, 1, 6, 0],
               [5, 192, 4, 1, 6, 0],
               [5, 192, 4, 1, 6, 0],
               [3, 320, 4, 1, 6, 0]],
              [[1, 1280, 0, 1, 0, -1]]
              ],
        'e': [[[3, 32, 0, 2, 0, -1]],
              [[3, 24, 0, 1, 3, 1]],
              [[3, 32, 0, 2, 8, 1],
               [3, 32, 0, 1, 8, 1]],
              [[3, 48, 0, 2, 8, 1],
               [3, 48, 0, 1, 8, 1],
               [3, 48, 0, 1, 8, 1],
               [3, 48, 0, 1, 8, 1]],
              [[5, 96, 0, 2, 8, 0],
               [5, 96, 0, 1, 8, 0],
               [5, 96, 0, 1, 8, 0],
               [5, 96, 0, 1, 8, 0],
               [5, 96, 0, 1, 8, 0],
               [5, 144, 0, 1, 8, 0],
               [5, 144, 0, 1, 8, 0],
               [5, 144, 0, 1, 8, 0],
               [5, 144, 0, 1, 8, 0]],
              [[5, 192, 0, 2, 8, 0],
               [5, 192, 0, 1, 8, 0]],
              [[1, 1280, 0, 1, 0, -1]]
              ]
    }  # yapf: disable

    # Parameters to build different kinds of architecture.
    # From left to right: scaling factor for width, scaling factor for depth,
    # resolution.
    arch_settings = {
        'b0': (1.0, 1.0, 224),
        'b1': (1.0, 1.1, 240),
        'b2': (1.1, 1.2, 260),
        'b3': (1.2, 1.4, 300),
        'b4': (1.4, 1.8, 380),
        'b5': (1.6, 2.2, 456),
        'b6': (1.8, 2.6, 528),
        'b7': (2.0, 3.1, 600),
        'b8': (2.2, 3.6, 672),
        'es': (1.0, 1.0, 224),
        'em': (1.0, 1.1, 240),
        'el': (1.2, 1.4, 300)
    }

    def __init__(self,
                 arch='b0',
                 drop_path_rate=0.,
                 out_indices=(6, ),
                 frozen_stages=0,
                 conv_cfg=dict(type='Conv2dAdaptivePadding'),
                 norm_cfg=dict(type='BN', eps=1e-3),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(
                         type='Constant',
                         layer=['_BatchNorm', 'GroupNorm'],
                         val=1)
                 ]):
        super(EfficientNet, self).__init__(init_cfg)
        assert arch in self.arch_settings, \
            f'"{arch}" is not one of the arch_settings ' \
            f'({", ".join(self.arch_settings.keys())})'
        self.arch_setting = self.arch_settings[arch]
        self.layer_setting = self.layer_settings[arch[:1]]
        for index in out_indices:
            if index not in range(0, len(self.layer_setting)):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, {len(self.layer_setting)}). '
                                 f'But received {index}')

        if frozen_stages not in range(len(self.layer_setting) + 1):
            raise ValueError('frozen_stages must be in range(0, '
                             f'{len(self.layer_setting) + 1}). '
                             f'But received {frozen_stages}')
        self.drop_path_rate = drop_path_rate
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.layer_setting = model_scaling(self.layer_setting,
                                           self.arch_setting)
        block_cfg_0 = self.layer_setting[0][0]
        block_cfg_last = self.layer_setting[-1][0]
        self.in_channels = make_divisible(block_cfg_0[1], 8)
        self.out_channels = block_cfg_last[1]
        self.layers = nn.ModuleList()
        self.layers.append(
            ConvModule(
                in_channels=3,
                out_channels=self.in_channels,
                kernel_size=block_cfg_0[0],
                stride=block_cfg_0[3],
                padding=block_cfg_0[0] // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.make_layer()
        # Avoid building unused layers in mmdetection.
        if len(self.layers) < max(self.out_indices) + 1:
            self.layers.append(
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=block_cfg_last[0],
                    stride=block_cfg_last[3],
                    padding=block_cfg_last[0] // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def make_layer(self):
        # Without the first and the final conv block.
        layer_setting = self.layer_setting[1:-1]

        total_num_blocks = sum([len(x) for x in layer_setting])
        block_idx = 0
        dpr = [
            x.item()
            for x in torch.linspace(0, self.drop_path_rate, total_num_blocks)
        ]  # stochastic depth decay rule

        for i, layer_cfg in enumerate(layer_setting):
            # Avoid building unused layers in mmdetection.
            if i > max(self.out_indices) - 1:
                break
            layer = []
            for i, block_cfg in enumerate(layer_cfg):
                (kernel_size, out_channels, se_ratio, stride, expand_ratio,
                 block_type) = block_cfg

                mid_channels = int(self.in_channels * expand_ratio)
                out_channels = make_divisible(out_channels, 8)
                if se_ratio <= 0:
                    se_cfg = None
                else:
                    # In mmdetection, the `divisor` is deleted to align
                    # the logic of SELayer with mmpretrain.
                    se_cfg = dict(
                        channels=mid_channels,
                        ratio=expand_ratio * se_ratio,
                        act_cfg=(self.act_cfg, dict(type='Sigmoid')))
                if block_type == 1:  # edge tpu
                    if i > 0 and expand_ratio == 3:
                        with_residual = False
                        expand_ratio = 4
                    else:
                        with_residual = True
                    mid_channels = int(self.in_channels * expand_ratio)
                    if se_cfg is not None:
                        # In mmdetection, the `divisor` is deleted to align
                        # the logic of SELayer with mmpretrain.
                        se_cfg = dict(
                            channels=mid_channels,
                            ratio=se_ratio * expand_ratio,
                            act_cfg=(self.act_cfg, dict(type='Sigmoid')))
                    block = partial(EdgeResidual, with_residual=with_residual)
                else:
                    block = InvertedResidual
                layer.append(
                    block(
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        mid_channels=mid_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        se_cfg=se_cfg,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        drop_path_rate=dpr[block_idx],
                        with_cp=self.with_cp,
                        # In mmdetection, `with_expand_conv` is set to align
                        # the logic of InvertedResidual with mmpretrain.
                        with_expand_conv=(mid_channels != self.in_channels)))
                self.in_channels = out_channels
                block_idx += 1
            self.layers.append(Sequential(*layer))

    def forward(self, x):
        #print(x.shape)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        #print(len(outs))
        #print(outs[0].shape,outs[1].shape,outs[2].shape,outs[3].shape,outs[4].shape)

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(EfficientNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

import torch
import torch.nn as nn

import timm

import os


@MODELS.register_module()
class EfficientNetB0Backbone(BaseModule):
    """EfficientNet-B0 backbone using timm.features_only + 本地权重加载.

    Args:
        out_indices (tuple[int]): 哪几个 stage 的特征输出。
            对于 efficientnet_b0.ra_in1k，features_only 输出 5 层，
            通道大致是 [16, 24, 40, 112, 320]。
        pretrained (bool): 是否让 timm 自己下载预训练权重。
            如果你给了 checkpoint_path，一般设为 False。
        checkpoint_path (str): 本地权重路径（HF 下载的 pytorch_model.bin）。
        in_chans (int): 输入通道数。
        init_cfg (dict | None): MMEngine 的初始化配置.
    """

    def __init__(
        self,
        out_indices=(1, 2, 3, 4),
        pretrained=False,
        checkpoint_path: str = '',
        in_chans=3,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.out_indices = out_indices
        self.in_chans = in_chans
        self.checkpoint_path = checkpoint_path

        # 如果没有给本地 ckpt，就按原来逻辑是否用 timm 自带 pretrained
        use_pretrained = pretrained and (not checkpoint_path)

        # 🔴 这里非常关键：
        # - 不再传 checkpoint_path 给 timm.create_model（避免 features_only + ckpt bug）
        # - 模型名直接用 HF 卡片上的名字：'efficientnet_b0.ra_in1k'
        self.backbone = timm.create_model(
            'efficientnet_b0.ra_in1k',
            pretrained=use_pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_chans,
        )

        # 从本地 ckpt 加权重（如果提供）
        if checkpoint_path:
            self._load_local_checkpoint(checkpoint_path)

        # 记录每个特征层的通道数
        self._feat_channels = self.backbone.feature_info.channels()

    # ------------------------------------------------------------------ #
    #  本地 ckpt 加载逻辑
    # ------------------------------------------------------------------ #
    def _load_local_checkpoint(self, checkpoint_path: str):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

        ckpt = torch.load(checkpoint_path, map_location='cpu')

        # HF 的 pytorch_model.bin 通常就是纯 state_dict
        # 但也兼容 dict 里包了一层 'state_dict' / 'model' 的情况
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            elif 'model' in ckpt:
                ckpt = ckpt['model']

        incompatible = self.backbone.load_state_dict(ckpt, strict=False)

        missing_keys = incompatible.missing_keys
        unexpected_keys = incompatible.unexpected_keys

        # 不必须，但可以打印一下方便你自己检查
        if missing_keys:
            print(f'[EfficientNetB0Backbone] missing keys: {len(missing_keys)} '
                  f'(e.g. {missing_keys[:5]})')
        if unexpected_keys:
            print(f'[EfficientNetB0Backbone] unexpected keys: {len(unexpected_keys)} '
                  f'(e.g. {unexpected_keys[:5]})')

    # ------------------------------------------------------------------ #
    #  MMDet 需要知道每个输出层的通道数
    # ------------------------------------------------------------------ #
    @property
    def out_channels(self):
        return [self._feat_channels[i] for i in self.out_indices]

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)  # timm 返回 list[Tensor]
        return tuple(feats)

