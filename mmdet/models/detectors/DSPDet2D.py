from mmdet.structures import DetDataSample
from mmdet.registry import MODELS
from mmdet.models.detectors.base import BaseDetector
from typing import List, Tuple, Union
from torch import Tensor
import torch
from mmdet.utils import OptConfigType, OptMultiConfig
@MODELS.register_module()
class DSPDet2D(BaseDetector):
    """2D Single Stage Detector with Progressive Feature Pruning.

    Args:
        backbone (dict): Config of the backbone.
        head (dict): Config of the head.
        neck (dict, optional): Config of the neck. Defaults to None.
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
        pretrained (str, optional): Deprecated initialization parameter.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None,
                 data_preprocessor: OptConfigType = None):
        super(DSPDet2D, self).__init__(init_cfg=init_cfg,data_preprocessor=data_preprocessor)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = MODELS.build(head)
        self.init_weights()

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features from images.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).

        Returns:
            tuple[Tensor]: Multi-level features from the backbone + neck.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, inputs: Tensor, data_samples: List[DetDataSample]) -> dict:
        """新版接口：从 data_samples 解析标注"""
        # 提取 gt_bboxes 和 gt_labels
        gt_bboxes = [sample.gt_instances.bboxes for sample in data_samples]
        gt_labels = [sample.gt_instances.labels for sample in data_samples]
        #DSPNet.loss inputs:
        #Tensor shape: torch.Size([1, 3, 512, 512])

        # 提取 img_metas（新版存储在 metainfo 中）
        img_metas = [sample.metainfo for sample in data_samples]
        #print("DSPNet.loss inputs:")
        #x=inputs
        #print(f"Tensor shape: {x.shape}") if isinstance(x, torch.Tensor) else print(f"Tuple len: {len(x)}, shapes: {[t.shape if hasattr(t, 'shape') else type(t) for t in x]}")
        # 调用特征提取和头部计算
        x = self.extract_feat(inputs)
        #x Tuple len: 4,
        # shapes: [torch.Size([1, 64, 256, 256]), torch.Size([1, 128, 128, 128]), torch.Size([1, 128, 64, 64]), torch.Size([1, 128, 32, 32])]
        #print("DSPNet.loss inputs:")
        #print(f"Tensor shape: {x.shape}") if isinstance(x, torch.Tensor) else print(f"Tuple len: {len(x)}, shapes: {[t.shape if hasattr(t, 'shape') else type(t) for t in x]}")

        losses = self.head.forward_train(x, gt_bboxes, gt_labels, img_metas)
        return losses

    def predict(self, img, img_metas, **kwargs):
    #def predict(self, img, img_metas, **kwargs):
        """Test without augmentation.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            list[dict]: Predicted 2d boxes.
        """
        x = self.extract_feat(img)
        detections = self.head.forward_test(x, img_metas)

        # Convert to mmdet expected format
        results = []
        for boxes, scores, labels in detections:
            results.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
        return results

    def simple_test(self, img, img_metas, **kwargs):
        """Test without augmentation.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            list[dict]: Predicted 2d boxes.
        """
        x = self.extract_feat(img)
        detections = self.head.forward_test(x, img_metas)

        # Convert to mmdet expected format
        results = []
        for boxes, scores, labels in detections:
            results.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            imgs (list[Tensor]): List of augmented images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            list[dict]: Predicted 2d boxes.
        """
        raise NotImplementedError("Augmented testing not implemented for DSPDet2D")

    def _forward(self, img):
        """Used for computing network flops.

        Args:
            img (Tensor): Input images.

        Returns:
            dict: Contains feature maps from head.
        """
        x = self.extract_feat(img)
        outs = self.head(x)
        return outs