import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import bias_init_with_prob
from mmcv.ops import nms
from mmengine.model import BaseModule
from mmdet.registry import MODELS

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_losses(gt_bboxes, points_flattened, final_tensor, bbox_pred):
    """
    计算分类损失和回归损失
    Args:
        gt_bboxes: [m, 4], 真实 bbox (x1, y1, x2, y2)
        points_flattened: [87040, 2], 所有点的中心坐标 (cx, cy)
        final_tensor: [87040, 1], 二值掩码 (0=背景, 1=前景)
        bbox_pred: [87040, 4], 预测的 bbox 偏移量 (dx, dy, dw, dh)
    Returns:
        cls_loss: 分类损失（Focal Loss）
        bbox_loss: 回归损失（Smooth L1 Loss）
    """
    device = gt_bboxes.device

    # --- 1. 筛选正样本（final_tensor == 1的点） ---
    pos_mask = (final_tensor.squeeze(-1) == 1)  # [87040]
    pos_points = points_flattened[pos_mask]  # [K, 2], K=正样本数
    pos_bbox_pred = bbox_pred[pos_mask]  # [K, 4]

    if len(pos_points) == 0:
        #print(len(pos_points))
        #print(pos_points.shape)
        #print(pos_mask.sum())
        #print(gt_bboxes.shape)
        #print("??????????????????????????????????")
        # 无正样本时返回零损失（或跳过回归损失）
        return torch.tensor(0.0, device=device)

    # --- 2. 匹配正样本点到最近的 gt_bbox ---
    # 计算所有正样本点到 gt_bboxes 中心的 L1 距离
    gt_centers = torch.stack([
        (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2,  # cx
        (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2  # cy
    ], dim=1)  # [m, 2]

    distances = torch.cdist(pos_points, gt_centers, p=1)  # [K, m]
    matched_gt_indices = distances.argmin(dim=1)  # [K], 每个点匹配的 gt 索引
    matched_gt_boxes = gt_bboxes[matched_gt_indices]  # [K, 4]

    # --- 3. 解码预测 bbox ---
    # 预测的 bbox: [cx + dx, cy + dy, exp(dw), exp(dh)]
    pred_cx = pos_points[:, 0] + pos_bbox_pred[:, 0]  # cx + dx
    pred_cy = pos_points[:, 1] + pos_bbox_pred[:, 1]  # cy + dy
    pred_w = torch.exp(pos_bbox_pred[:, 2])  # exp(dw)
    pred_h = torch.exp(pos_bbox_pred[:, 3])  # exp(dh)

    pred_boxes = torch.stack([
        pred_cx - 0.5 * pred_w,  # x1
        pred_cy - 0.5 * pred_h,  # y1
        pred_cx + 0.5 * pred_w,  # x2
        pred_cy + 0.5 * pred_h  # y2
    ], dim=1)  # [K, 4]

    # --- 4. 计算回归损失（Smooth L1）---
    bbox_loss = nn.SmoothL1Loss()(pred_boxes, matched_gt_boxes)

    return bbox_loss
@MODELS.register_module()
class DSPHeadImage2(BaseModule):
    def __init__(self,
                 n_classes,
                 in_channels,
                 out_channels,
                 n_reg_outs,
                 assigner,
                 prune_threshold=0.5,
                 bbox_loss=dict(type='IoULoss', reduction='none'),
                 cls_loss=dict(type='L2Loss', reduction='mean'),
                 keep_loss=dict(type='L2Loss', reduction='mean'),
                 train_cfg=None,
                 test_cfg=None):
        super(DSPHeadImage2, self).__init__()
        self.prune_threshold = prune_threshold
        self.assigner = MODELS.build(assigner)
        self.bbox_loss = MODELS.build(bbox_loss)
        self.loss_cls = MODELS.build(cls_loss)
        self.keep_loss = MODELS.build(keep_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes=1
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)


    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        # 检测头
        self.bbox_conv = nn.Conv2d(out_channels, n_reg_outs, kernel_size=1)
        self.cls_conv = nn.Conv2d(out_channels, n_classes, kernel_size=1)

        # 剪枝头（更深的可学习结构）
        self.keep_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64 if _ == 0 else out_channels, out_channels//2, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels//2, out_channels // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, 1, kernel_size=1)
            ) for _ in range(len(in_channels))
        ])

        # 特征处理块
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self._make_up_block(in_channels[i], in_channels[i - 1]))

            self.__setattr__(
                f'lateral_block_{i}',
                self._make_block(in_channels[i], in_channels[i]))
            self.__setattr__(
                f'out_block_{i}',
                self._make_block(in_channels[i], out_channels))

    def _make_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def _make_up_block(self, in_channels, out_channels):#64->128
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def init_weights(self):
        # 检测头初始化
        nn.init.normal_(self.bbox_conv.weight, std=.01)
        nn.init.normal_(self.cls_conv.weight, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

        # 剪枝头初始化
        for conv_seq in self.keep_conv:
            for layer in conv_seq:
                if isinstance(layer, nn.Conv2d):
                    if layer.kernel_size == (1, 1):  # 最后一层
                        nn.init.normal_(layer.weight, std=.01)
                    else:
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

        # 其他层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m not in self.keep_conv and m != self.bbox_conv and m != self.cls_conv:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_single(self, x):
        """单层级前向传播"""
        #print(f"输入x的统计: min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}, NaN={torch.isnan(x).any().item()}")

        bbox_pred = self.bbox_conv(x)
        #print("bbox+pred",bbox_pred.shape)
        #print(f"bbox_pred的统计: min={bbox_pred.min().item():.6f}, max={bbox_pred.max().item():.6f}, NaN={torch.isnan(bbox_pred).any().item()}")

        cls_pred = self.cls_conv(x)

        # 生成坐标网格
        batch_size = x.shape[0]
        h, w = x.shape[-2:]

        # 生成基础坐标网格 (H, W, 2)
        y_coords = torch.arange(h, device=x.device, dtype=torch.float32).view(-1, 1).expand(h, w)
        x_coords = torch.arange(w, device=x.device, dtype=torch.float32).view(1, -1).expand(h, w)
        grid = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)

        # 扩展到batch维度 (batch, H*W, 2)
        points = grid.reshape(1, -1, 2).expand(batch_size, -1, -1)  # -1表示自动推导
        #print("points",points.shape)
        #print(bbox_pred.shape)
        return bbox_pred, cls_pred, points

    def _get_gt_keep_mask(self, x, gt_bboxes_list, img_metas):
        batch_size, _, h, w = x.shape
        device = x.device
        masks = []

        # 预先生成网格坐标（非原地）
        yy, xx = torch.meshgrid(torch.arange(h, device=device),
                                torch.arange(w, device=device), indexing='ij')
        xx = xx.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        yy = yy.unsqueeze(0).unsqueeze(0)

        for i in range(batch_size):
            # 使用全零初始化（确保非原地）
            mask = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32, requires_grad=False)

            gt_bboxes = gt_bboxes_list[i] if isinstance(gt_bboxes_list, (list, tuple)) else gt_bboxes_list
            img_meta = img_metas[i] if isinstance(img_metas, (list, tuple)) else img_metas

            if isinstance(img_meta, dict):
                img_shape = img_meta.get('img_shape', (1, 1))
            else:
                img_shape = (1, 1)

            if len(gt_bboxes) > 0:
                img_h, img_w = img_shape[:2]
                scale_x = w / max(img_w, 1)
                scale_y = h / max(img_h, 1)

                # 一次性处理所有bbox
                combined_mask = torch.zeros_like(mask)
                for bbox in gt_bboxes:
                    x1 = int(bbox[0] * scale_x)
                    y1 = int(bbox[1] * scale_y)
                    x2 = int(bbox[2] * scale_x)
                    y2 = int(bbox[3] * scale_y)

                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(x1 + 1, min(x2, w - 1))
                    y2 = max(y1 + 1, min(y2, h - 1))

                    # 使用非原地操作创建区域掩码
                    region_mask = ((xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)).float()
                    combined_mask = combined_mask + region_mask  # 非原地加法

                mask = (combined_mask > 0).float()  # 二值化

            masks.append(mask)

        return torch.cat(masks, dim=0)

    def _get_gt_keep_mask_old(self, x, gt_bboxes_list, img_metas):
        """根据GT框生成剪枝掩码（支持batch处理）

        Args:
            x (Tensor): 输入特征图，形状为(B, C, H, W)
            gt_bboxes_list (list[Tensor]): 每个样本的GT框列表，长度为batch_size
            img_metas (list[dict]): 每个样本的meta信息列表，长度为batch_size

        Returns:
            Tensor: 剪枝掩码，形状为(B, 1, H, W)
        """
        batch_size, _, h, w = x.shape
        device = x.device
        masks = []
        #print("img_metas",img_metas)

        for i in range(batch_size):
            # 为当前样本创建空掩码
            mask = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)

            # 获取当前样本的GT框和meta信息
            gt_bboxes = gt_bboxes_list[i] if isinstance(gt_bboxes_list, (list, tuple)) else gt_bboxes_list
            img_meta = img_metas[i] if isinstance(img_metas, (list, tuple)) else img_metas

            # 如果img_metas是字典列表，确保我们能获取到img_shape
            if isinstance(img_meta, dict):
                img_shape = img_meta.get('img_shape', (1, 1))
            else:
                img_shape = (1, 1)  # 默认值

            if len(gt_bboxes) > 0:
                # 计算缩放因子
                img_h, img_w = img_shape[:2]
                scale_x = w / max(img_w, 1)  # 避免除以0
                scale_y = h / max(img_h, 1)

                # 转换GT框到特征图尺度
                for bbox in gt_bboxes:
                    x1 = int(bbox[0] * scale_x)
                    y1 = int(bbox[1] * scale_y)
                    x2 = int(bbox[2] * scale_x)
                    y2 = int(bbox[3] * scale_y)

                    # 确保不越界
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(x1 + 1, min(x2, w - 1))
                    y2 = max(y1 + 1, min(y2, h - 1))

                    mask[0, 0, y1:y2, x1:x2] = 1

            masks.append(mask)

        # 合并batch中的掩码
        return torch.cat(masks, dim=0)

    def forward(self, x_tuple, gt_bboxes=None, gt_labels=None, img_metas=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 确保输入不需要梯度
        if gt_bboxes is not None:
            gt_bboxes = [bbox.detach() if bbox.requires_grad else bbox for bbox in gt_bboxes]

        keep_gts1 = []
        # 生成掩码时禁用梯度
        if self.training:
            with torch.no_grad():
                # 目标分辨率
                resolutions = [(64, 64), (128, 128), (256, 256)]


                for h, w in resolutions:
                    masks = []
                    scale_x = w / 512.0  # 原图是 512x512，缩放到目标分辨率
                    scale_y = h / 512.0

                    for bboxes in gt_bboxes:  # 遍历每张图片的 bboxes
                        # 直接在 GPU 上初始化全 0 掩码
                        mask = torch.zeros((1, h, w), dtype=torch.float32, device=device)

                        for bbox in bboxes:  # 遍历每个 bbox
                            # 缩放到目标分辨率（直接在 GPU 上计算）
                            x1, y1, x2, y2 = bbox * torch.tensor([scale_x, scale_y, scale_x, scale_y], device=device)

                            # 确保坐标在有效范围内（转换为整数）
                            x1, y1 = max(0, int(x1)), max(0, int(y1))
                            x2, y2 = min(w, int(x2)), min(h, int(y2))

                            if x2 > x1 and y2 > y1:  # 确保 bbox 有效
                                mask[:, y1:y2, x1:x2] = 1  # 将 bbox 区域设为 1

                        masks.append(mask.unsqueeze(0))  # 添加 batch 维度

                    # 合并所有图片的掩码（结果张量自动在 GPU 上）
                    keep_gts1.append(torch.cat(masks, dim=0))



        # 验证输出形状和设备
        #for mask in keep_gts1:
        #    print(f"Shape: {mask.shape}, Device: {mask.device}")

        # 验证输出形状
        #print([mask.shape for mask in keep_gts1])
        #print("?????????????????????????????")

        """完整前向传播"""
        #print("gt_bbox::",len(gt_bboxes),gt_bboxes[0].shape)
        torch.autograd.set_detect_anomaly(True)  # 放在代码最开始
        bbox_preds, cls_preds, points = [], [], []
        keep_preds=[]#, keep_gts1 = , []

        # 从最低分辨率开始处理
        #print("len(x_tuple)",len(x_tuple))
        x = x_tuple[-1]


        if self.training:
            # 处理当前层特征
            x = self.__getattr__(f'lateral_block_{len(x_tuple) - 1}')(x)
            # print("after lateral", len(x_tuple) - 1, x.shape)
            out = self.__getattr__(f'out_block_{len(x_tuple) - 1}')(x)
            # print("after out", len(x_tuple) - 1, out.shape)

            # 获取检测预测
            bbox_pred, cls_pred, point = self._forward_single(out)
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)

            keep_pred = self.keep_conv[len(x_tuple) - 1](x)
            # print("after keep:", keep_pred.shape)
            # print("keep_pred:",keep_pred.shape)
            keep_preds.append(keep_pred)

            for i in range(len(x_tuple) - 1, -1, -1):
                if i < len(x_tuple) - 1:
                    # print("ii",i)
                    # 上采样并与高分辨率特征融合
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    #print(x.shape)
                    x = F.interpolate(x, size=x_tuple[i].shape[-2:],
                                      mode='bilinear', align_corners=False)
                    #print(x_tuple[i].shape)
                    #print(x.shape)
                    x = torch.add(x, self.__getattr__(f'lateral_block_{i}')(x_tuple[i]))
                    #print("before prun:", x.shape)
                    keep_pred = self.keep_conv[i](x)
                    #print("after keep:", keep_pred.shape)
                    # print("keep_pred:",keep_pred.shape)
                    keep_preds.append(keep_pred)

                    x = x * keep_gts1[2 - i].clone()

                    # 推理时使用预测剪枝
                    if keep_preds:
                        keep_pred = keep_preds[-1]
                        #print("keep_pred", keep_pred.shape)
                        prune_mask = (keep_pred.sigmoid() > self.prune_threshold).float()

                    out = self.__getattr__(f'out_block_{i}')(x)
                    #print("after out", i, out.shape)

                    bbox_pred, cls_pred, point = self._forward_single(out)
                    bbox_preds.append(bbox_pred)
                    cls_preds.append(cls_pred)
                    points.append(point)

        else:
            # 处理当前层特征
            x = self.__getattr__(f'lateral_block_{len(x_tuple) - 1}')(x)
            #print("after lateral", len(x_tuple) - 1, x.shape)
            out = self.__getattr__(f'out_block_{len(x_tuple) - 1}')(x)
            #print("after out", len(x_tuple) - 1, out.shape)

            # 获取检测预测
            bbox_pred, cls_pred, point = self._forward_single(out)
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)

            keep_pred = self.keep_conv[len(x_tuple) - 1](x)
            #print("after keep:", keep_pred.shape)
            # print("keep_pred:",keep_pred.shape)
            keep_preds.append(keep_pred)

            for i in range(len(x_tuple) - 1, -1, -1):
                if i < len(x_tuple) - 1:
                    # print("ii",i)
                    # 上采样并与高分辨率特征融合
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    x = F.interpolate(x, size=x_tuple[i].shape[-2:],
                                      mode='bilinear', align_corners=False)
                    x = torch.add(x, self.__getattr__(f'lateral_block_{i}')(x_tuple[i]))
                    #print("before prun:", x.shape)
                    keep_pred = self.keep_conv[i](x)
                    #print("after keep:", keep_pred.shape)
                    # print("keep_pred:",keep_pred.shape)
                    keep_preds.append(keep_pred)

                    # 推理时使用预测剪枝
                    if keep_preds:
                        keep_pred = keep_preds[-1]
                        #print("keep_pred", keep_pred.shape)
                        prune_mask = (keep_pred.sigmoid() > self.prune_threshold).float()
                        x = torch.mul(x, prune_mask)

                    out = self.__getattr__(f'out_block_{i}')(x)
                    #print("after out", i, out.shape)

                    bbox_pred, cls_pred, point = self._forward_single(out)
                    bbox_preds.append(bbox_pred)
                    cls_preds.append(cls_pred)
                    points.append(point)

        # 反转列表使顺序从高到低分辨率
        #print("point_append:", bbox_preds[0].shape, bbox_preds[1].shape, bbox_preds[2].shape, bbox_preds[3].shape)
        #print("point_append:",points[0].shape,points[1].shape,points[2].shape,points[3].shape)
        return bbox_preds[::-1], cls_preds[::-1], points[::-1], keep_preds[::-1], keep_gts1[::-1]

    def _bbox_pred_to_bbox(self, points, bbox_pred):
        """将预测偏移量转换为绝对坐标"""
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        # 假设bbox_pred格式为[dx, dy, dw, dh]
        x_center = points[..., 0] + bbox_pred[..., 0]
        y_center = points[..., 1] + bbox_pred[..., 1]
        w = bbox_pred[..., 2].exp()  # 使用exp保证宽度为正
        h = bbox_pred[..., 3].exp()

        return torch.stack([x_center, y_center, w, h], dim=-1)

    def _bbox_to_loss(self, bbox):
        """将bbox转换为损失计算格式[x1,y1,x2,y2]"""
        return torch.stack([
            bbox[..., 0] - bbox[..., 2] / 2,
            bbox[..., 1] - bbox[..., 3] / 2,
            bbox[..., 0] + bbox[..., 2] / 2,
            bbox[..., 1] + bbox[..., 3] / 2
        ], dim=-1)

    def _loss_single(self, bbox_preds, cls_preds, points, gt_bboxes, gt_labels, img_meta):
        torch.autograd.set_detect_anomaly(True)  # 放在代码最开始
        #print("????????????????????????????")
        #print(gt_bboxes,"  ",gt_bboxes.shape[0])
        # 检查 gt_bboxes 是否为空
        if gt_bboxes.shape[0] == 0:
            device = bbox_preds[0].device
            total_points = sum(p.shape[0] for p in points)
            return torch.tensor(0.0, device=device),torch.tensor(0.0, device=device),torch.zeros(total_points, dtype=torch.bool, device=device)
        mode = "dele1"
        # Step 1: 处理 "dele" 模式
        if mode == "dele":
            pre_mask = []
            for bbox_pred in bbox_preds:
                #print(bbox_pred.shape)
                #print("??????????????????")
                _, h, w = bbox_pred.shape[-3:]
                level_mask = torch.zeros((1, h, w), device=bbox_pred.device)
                for bbox in gt_bboxes:
                    x1, y1, x2, y2 = (bbox * torch.tensor([w, h, w, h], device=bbox.device)).int()
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        level_mask[0, y1:y2, x1:x2] = 1
                pre_mask.append(level_mask)

            # 应用 pre_mask
            new_points = []
            for pt, mask in zip(points, pre_mask):
                flat_mask = mask.view(-1).bool()
                if flat_mask.sum() > 0:
                    new_points.append(pt[flat_mask])

            if not new_points:  # 所有点被过滤
                device = bbox_preds[0].device
                total_points = sum(p.shape[0] for p in points)
                return (
                    torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device),
                    torch.zeros(total_points, dtype=torch.bool, device=device)
                )
            points = new_points

        points_resized = [points_level * (512 // int((points_level.shape[0]) ** 0.5)) for points_level in points]
        # Step 2: 展平预测和点
        points_flattened = torch.cat(points_resized)
        #print("points",points_resized)
        #print("gt_bboxes", gt_bboxes)
        #print("gt_labels", gt_labels)
        #print("bbox_pred", bbox_preds)
        #print("cls_pred", cls_preds)
        bbox_pred = torch.cat([p.permute(1,2,0).reshape(-1, 4) for p in bbox_preds])
        cls_pred = torch.cat([p.permute(1,2,0).reshape(-1, self.num_classes) for p in cls_preds])
        #print("bbox_pred", bbox_pred)
        #print("gt_bboxes shape:", gt_bboxes.shape)
        mask = torch.zeros((512, 512), device=gt_bboxes.device)
        x1, y1, x2, y2 = gt_bboxes.reshape(-1, 4).permute(1, 0).round().long()  # shape [4, m]
        x1 = x1.clamp(min=0, max=511)
        y1 = y1.clamp(min=0, max=511)
        x2 = x2.clamp(min=0, max=511)
        y2 = y2.clamp(min=0, max=511)
        for i in range(gt_bboxes.reshape(-1, 4).permute(1, 0).shape[1]):
            # 创建新张量而不是原地修改
            new_mask = torch.zeros_like(mask)
            new_mask[y1[i]:y2[i] + 1, x1[i]:x2[i] + 1] = 1
            mask = mask + new_mask  # 使用加法而不是直接赋值
        mask = (mask > 0).float()  # 确保最终是二值掩码

        ##mask = torch.zeros((512, 512), device=gt_bboxes.device)
        ##x1, y1, x2, y2 = gt_bboxes.reshape(-1, 4).permute(1, 0).round().long()  # shape [4, m]
        ##x1 = x1.clamp(min=0, max=511)
        ##y1 = y1.clamp(min=0, max=511)
        ##x2 = x2.clamp(min=0, max=511)
        ##y2 = y2.clamp(min=0, max=511)
        ##for i in range(gt_bboxes.reshape(-1, 4).permute(1, 0).shape[1]):
        ##    mask[y1[i]:y2[i] + 1, x1[i]:x2[i] + 1] = 1

        def downsample_mask(mask, scale_factor):
            """下采样并添加通道维度"""
            mask_4d = mask.unsqueeze(0).unsqueeze(0).float()  # [1,1,512,512]
            downsampled = F.max_pool2d(mask_4d, kernel_size=scale_factor, stride=scale_factor)
            #downsampled = F.avg_pool2d(mask_4d, kernel_size=scale_factor, stride=scale_factor)
            downsampled = (downsampled.squeeze() > 0.5).float()  # 二值化
            return downsampled.unsqueeze(-1)  # [H,W,1]

        # 生成4个尺度的掩码
        mask_256 = downsample_mask(mask, scale_factor=2)  # [256,256,1]
        mask_128 = downsample_mask(mask, scale_factor=4)  # [128,128,1]
        mask_64 = downsample_mask(mask, scale_factor=8)  # [64,64,1]
        mask_32 = downsample_mask(mask, scale_factor=16)  # [32,32,1]

        flattened_masks = [
            mask_256.reshape(-1, 1),  # [256 * 256, 1]
            mask_128.reshape(-1, 1),  # [128 * 128, 1]
            mask_64.reshape(-1, 1),  # [64 * 64, 1]
            mask_32.reshape(-1, 1)  # [32 * 32, 1]
        ]

        # 沿第0维度拼接（最终形状为 [N, 1]，N=256²+128²+64²+32²）
        final_tensor = torch.cat(flattened_masks, dim=0)

        #print("howmany_mask:", mask.sum().item())
        # Step 3: 分配真实框
        assigned_ids = self.assigner.assign(points_resized, gt_bboxes, gt_labels, img_meta)
        #cls_targets=assigned_ids+1
        pos_mask = assigned_ids >= 0
        #print("howmany:",final_tensor.sum().item())
        #print(cls_pred.max(),cls_pred.min())
        #print(cls_targets.max(),cls_targets.min())
        # Step 4: 计算分类损失（严格按要求的格式）
        cls_pred_normalized = -torch.sigmoid(cls_pred)
        #print(cls_pred_normalized.max(), cls_pred_normalized.min())
        cls_loss = self.loss_cls(cls_pred_normalized, final_tensor)
        #print("L1:",cls_loss,final_tensor.shape)

        #Step 5: 计算回归损失（严格按要求的格式）

        ###pos_points = points_flattened[pos_mask]  # [K,2]
        ###pos_bbox_pred = bbox_pred[pos_mask]  # [K,4]
        #print(pos_points.shape)
        #print(pos_bbox_pred.shape)
        #print("gt_bboxes",gt_bboxes.shape,gt_bboxes)
        #print("points_flattened", points_flattened.shape, points_flattened)
        #print("bbox_pred",bbox_pred.shape, bbox_pred)
        #print("final_tensor", final_tensor.shape, final_tensor)
        bbox_loss=compute_losses(gt_bboxes, points_flattened, final_tensor, bbox_pred)
        #pos_bbox_targets = gt_bboxes[assigned_ids[pos_mask]]/512  # [K,4]
        #print(pos_bbox_targets.shape)

         # 严格按要求的损失计算方式
        #pred_boxes = self._bbox_pred_to_bbox(pos_points, pos_bbox_pred)
        #target_boxes = self._bbox_pred_to_bbox(pos_points, pos_bbox_targets)
        #bbox_loss = self.bbox_loss(
        #    self._bbox_to_loss(pred_boxes),
        #    self._bbox_to_loss(target_boxes)
        #)
        #print("L2:",bbox_loss)
        #print(f"bbox_loss type564: {type(bbox_loss)}")

        return bbox_loss, cls_loss, pos_mask

    def _loss_single1(self, bbox_preds, cls_preds, points, gt_bboxes, gt_labels, img_meta):
        """单样本损失计算"""
        # 展平预测结果
        #print("==================This is the input of loss single======================")
        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
        #print(len(assigned_ids))
        #print("bbox_pred:",len(bbox_preds))
        #print(bbox_preds[0].shape,bbox_preds[1].shape,bbox_preds[2].shape,bbox_preds[3].shape)
        #print("cls_pred:",len(cls_preds))
        #print(cls_preds[0].shape,cls_preds[1].shape,cls_preds[2].shape,cls_preds[3].shape)

        bbox_preds = [p if p.dim() == 4 else p.unsqueeze(0) for p in bbox_preds]
        cls_preds = [p if p.dim() == 4 else p.unsqueeze(0) for p in cls_preds]
        bbox_pred = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, 4) for p in bbox_preds])
        cls_pred = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, cls_preds[0].shape[1]) for p in cls_preds])
        #points = torch.cat(points)
        #print("assigner_input",len(points))
        #print(points[0].shape,points[1].shape,points[2].shape,points[3].shape)
        #print("gt_bbox",len(gt_bboxes))
        #if len(gt_bboxes)!=0:
        #    print(gt_bboxes[0])
        #else:
        #    print(gt_bboxes)
        #print("gt_lables",len(gt_labels))
        #if len(gt_labels)!=0:
        #    print(gt_labels[0])
        #else:
        #    print(gt_labels)


        # 分配GT到预测点

        #print("==================This is the input of cls loss======================")

        #print("cls_pred", len(cls_pred))
        # 分类损失
        n_classes = cls_pred.shape[1]
        pos_mask = assigned_ids >= 0
        #print("n_classes",n_classes)
        if gt_labels.shape[0] == 0:
            cls_targets = torch.full_like(assigned_ids, n_classes)
        else:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        #print("cls_pred",cls_pred.shape)
        #print("cls_targets",cls_targets.shape)
        cls_loss = self.loss_cls(cls_pred, cls_targets)
        #print("cls_loss", cls_loss)
        #print("==================This is the input of bbox loss======================")

        # 回归损失
        bbox_loss = None
        #print(pos_mask.shape)
        #print("bbox_pred",bbox_preds[0].shape)
        #print(pos_mask.shape)
        if pos_mask.sum() > 0:
            pos_bbox_pred = bbox_pred[pos_mask]
        #    print("pos_mask", pos_mask[0])
        #    print("pos_mask dtype:", pos_mask.dtype)  # 应为 torch.bool 或 torch.long
        #    print("pos_mask shape:", pos_mask.shape)  # 应与 points 的第一维匹配
        #    print("points shape:", points[0].shape)  # 检查 points 的实际形状
            pos_points = points[pos_mask]
            pos_bbox_targets = gt_bboxes[assigned_ids][pos_mask]

            # 转换预测和目标格式
            pred_boxes = self._bbox_pred_to_bbox(pos_points, pos_bbox_pred)
            target_boxes = self._bbox_pred_to_bbox(pos_points, pos_bbox_targets)

            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(pred_boxes),
                self._bbox_to_loss(target_boxes))
            #print(f"bbox_loss type637: {type(bbox_loss)}")

        return bbox_loss, cls_loss, pos_mask

    def _loss(self, bbox_preds, cls_preds, points, gt_bboxes, gt_labels, img_metas, keep_preds, keep_gts):
        """总损失计算"""
        torch.autograd.set_detect_anomaly(True)  # 放在代码最开始
        bbox_losses, cls_losses, pos_masks = [], [], []
        keep_losses = []
        #print("loss_point:",points[0].shape)
        #print(len(bbox_preds))

        # 剪枝损失（仅训练时）
        if self.training:
            for pred, gt in zip(keep_preds, keep_gts):
                # 调整维度: pred (1,1,H,W) -> (H*W,), gt (1,1,H,W) -> (H*W,)
                #print(pred.shape, gt.shape)
                pred = pred.view(-1)
                gt = gt.view(-1)#.long()
                # 只计算有目标的区域（避免背景主导）
                #if gt.sum() > 0:

                loss = self.keep_loss(pred, gt)
                keep_losses.append(loss)

            keep_loss = torch.mean(torch.stack(keep_losses)) if keep_losses else 0
        else:
            keep_loss = 0
        #print("L3:",keep_loss)

        # 检测损失
        for i in range(len(img_metas)):
            bbox_loss, cls_loss, pos_mask = self._loss_single(
                bbox_preds=[p[i] for p in bbox_preds],
                cls_preds=[p[i] for p in cls_preds],
                points=[p[i] for p in points],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                img_meta=img_metas[i])

            if bbox_loss is not None:
                #print(f"bbox_loss type: {type(bbox_loss)}")
                bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            #pos_masks.append(pos_mask)
            #print(bbox_losses)

        return dict(
            bbox_loss=0.05*torch.mean(torch.stack(bbox_losses)) if bbox_losses else 0,
            cls_loss=100*torch.mean(torch.stack(cls_losses)) if cls_losses else 0,
            keep_loss=10 * keep_loss)  # 剪枝损失权重

    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        """训练前向传播"""
        bbox_preds, cls_preds, points, keep_preds, keep_gts = self(
            x, gt_bboxes, gt_labels, img_metas)
        return self._loss(bbox_preds, cls_preds, points, gt_bboxes, gt_labels, img_metas, keep_preds, keep_gts)

    def forward_test(self, x, img_metas):
        """简化测试流程"""
        bbox_preds, cls_preds, points, keep_preds, _ = self(
            x, None, None, img_metas)

        # 仅处理第一个样本（假设batch_size=1）
        detections = self._get_bboxes_single(
            bbox_preds, cls_preds, points, img_metas[0])

        return [detections]

    def _get_bboxes_single(self, bbox_preds, cls_preds, points, img_meta):
        """获取单样本最终检测框"""
        # 展平预测结果
        bbox_pred = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, 4) for p in bbox_preds])
        cls_pred = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, cls_preds[0].shape[1]) for p in cls_preds])
        #print(len(bbox_preds))
        #print(len(cls_preds))
        #print(bbox_preds[0].shape,bbox_preds[1].shape,bbox_preds[2].shape,bbox_preds[3].shape,bbox_preds[4].shape)
        #print(cls_preds[0].shape,cls_preds[1].shape,cls_preds[2].shape,cls_preds[3].shape,cls_preds[4].shape)
        scores = cls_pred.sigmoid()
        #print(scores.shape)
        #print(len(points))
        #print(points[0].shape, points[1].shape, points[2].shape, points[3].shape)
        points = torch.cat([p.squeeze(0) for p in points])

        # 应用sigmoid获取分类分数


        # 预过滤低分预测
        if self.test_cfg.get('nms_pre', 0) > 0:
            max_scores, _ = scores.max(dim=1)
            _, topk_inds = max_scores.topk(min(self.test_cfg.nms_pre, scores.shape[0]))
            bbox_pred = bbox_pred[topk_inds]
            scores = scores[topk_inds]
            points = points[topk_inds]

        # 转换预测框格式
        boxes = self._bbox_pred_to_bbox(points, bbox_pred)
        boxes = self._bbox_to_loss(boxes)  # 转换为x1,y1,x2,y2格式

        # 应用NMS
        dets = self._nms(boxes, scores, img_meta)
        return dets

    def _nms(self, boxes, scores, img_meta):
        """非极大值抑制"""
        n_classes = scores.shape[1]
        det_bboxes, det_scores, det_labels = [], [], []

        for cls_idx in range(n_classes):
            cls_scores = scores[:, cls_idx]
            valid_mask = cls_scores > self.test_cfg.score_thr
            if not valid_mask.any():
                continue

            cls_boxes = boxes[valid_mask]
            cls_scores = cls_scores[valid_mask]

            # 执行NMS
            keep = nms(cls_boxes, cls_scores, self.test_cfg.iou_thr)

            det_bboxes.append(cls_boxes[keep])
            det_scores.append(cls_scores[keep])
            det_labels.append(
                boxes.new_full((len(keep),), cls_idx, dtype=torch.long))

        if det_bboxes:
            det_bboxes = torch.cat(det_bboxes)
            det_scores = torch.cat(det_scores)
            det_labels = torch.cat(det_labels)
        else:
            det_bboxes = boxes.new_zeros((0, 4))
            det_scores = boxes.new_zeros((0,))
            det_labels = boxes.new_zeros((0,), dtype=torch.long)

        return det_bboxes, det_scores, det_labels