import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models.detectors.base import BaseDetector
from mmengine.model import BaseModule
from mmdet.registry import MODELS


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm_cfg=dict(type='BN')):
        super(BasicBlock2D, self).__init__()
        self.conv1 = build_conv_layer(
            dict(type='Conv2d'),
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = build_conv_layer(
            dict(type='Conv2d'),
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck2D(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm_cfg=dict(type='BN')):
        super(Bottleneck2D, self).__init__()
        self.conv1 = build_conv_layer(
            dict(type='Conv2d'),
            inplanes,
            planes,
            kernel_size=1,
            bias=False)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.conv2 = build_conv_layer(
            dict(type='Conv2d'),
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.conv3 = build_conv_layer(
            dict(type='Conv2d'),
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.bn3 = build_norm_layer(norm_cfg, planes * self.expansion)[1]
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@MODELS.register_module()
class DSPBackbone2D(BaseModule):
    """2D ResNet backbone with progressive feature pruning support.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input channels, 3 for RGB.
        num_stages (int, optional): Resnet stages. Default: 4.
        strides (Sequence[int], optional): Strides of the first block of each stage.
            Default: (1, 2, 2, 2).
        dilations (Sequence[int], optional): Dilation of each stage.
            Default: (1, 1, 1, 1).
        out_indices (Sequence[int], optional): Output from which stages.
            Default: (0, 1, 2, 3).
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """
    arch_settings = {
        18: (BasicBlock2D, (2, 2, 2, 2)),
        34: (BasicBlock2D, (3, 4, 6, 3)),
        50: (Bottleneck2D, (3, 4, 6, 3)),
        101: (Bottleneck2D, (3, 4, 23, 3)),
        152: (Bottleneck2D, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=True,
                 max_channels=None,
                 pool=True):
        super(DSPBackbone2D, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.style = style
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.max_channels = max_channels
        self.pool = pool
        self.out_indices = out_indices
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]

        self._make_stem_layer(in_channels)
        self.res_layers = []

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            if self.max_channels is not None:
                planes = min(planes, self.max_channels)

            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                norm_cfg=self.norm_cfg,
                with_cp=self.with_cp)

            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()
        self.feat_dim = self.block.expansion * 64 * 2 ** (
                len(self.stage_blocks) - 1)

    def _make_stem_layer(self, in_channels):
        self.conv1 = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1 = build_norm_layer(self.norm_cfg, 64)[1]
        self.relu = nn.ReLU(inplace=True)
        if self.pool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64

    def _make_res_layer(self,
                        block,
                        inplanes,
                        planes,
                        blocks,
                        stride=1,
                        dilation=1,
                        style='pytorch',
                        norm_cfg=None,
                        with_cp=False):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    dict(type='Conv2d'),
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1])

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                norm_cfg=norm_cfg))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(inplanes, planes, stride=1, norm_cfg=norm_cfg))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck2D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock2D):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        """Forward function."""
        #print(len(x))
        #print(x[0].shape)
        #print("??????????????????????????????????")
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.pool:
            x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        #print("backbone",len(outs),outs[0].shape,outs[1].shape,outs[2].shape,outs[3].shape)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super(DSPBackbone2D, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()