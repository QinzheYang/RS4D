import torch.nn as nn
import torch.nn.functional as F

from .fpn import FPN
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType

@MODELS.register_module()
class CBFPN(FPN):
    '''
    FPN with weight sharing
    which support mutliple outputs from cbnet
    '''

    def forward(self, inputs):
        #print(len(inputs))
        #print(len(inputs[0]),len(inputs[1]))
        if not isinstance(inputs[0], (list, tuple)):
            inputs = [inputs]
        if True:#self.training:
            #print("111111111111111111111")
            outs = []
            for x in inputs:
                out = super().forward(x)
                outs.append(out)

            # 假设 outs 是一个包含两个 tuple 的 tuple，每个 tuple 包含 5 个 tensor
            # 例如：outs = ((tensor1_1, tensor1_2, tensor1_3, tensor1_4, tensor1_5),
            #              (tensor2_1, tensor2_2, tensor2_3, tensor2_4, tensor2_5))

            # 创建一个新的 tuple，包含 5 个平均后的 tensor
            outnew = tuple(
                (outs[0][i] + outs[1][i]) / 2  # 对两个 tuple 中对应位置的 tensor 求平均
                for i in range(len(outs[0]))  # 遍历每个 tensor 的位置
            )

            return outnew
        else:
            print("222222222222222222")
            out = super().forward(inputs[-1])
            return out

    """
    if self.training:
        print("111111111111111111111")
        outs = []
        for x in inputs:
            out = super().forward(x)
            outs.append(out)
        print(outs)
        print(len(outs))
        print(len(outs[0]))
        print(outs[0][0].shape)#torch.Size([1, 256, 56, 56])
        print(outs[0][1].shape)#torch.Size([1, 256, 28, 28])
        print(outs[0][2].shape)#torch.Size([1, 256, 14, 14])
        print(outs[0][3].shape)#torch.Size([1, 256, 7, 7])
        print(outs[0][4].shape)#torch.Size([1, 256, 4, 4])
        return outs
    """
