import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import bias_init_with_prob
from mmcv.ops import nms
from mmengine.model import BaseModule
from mmdet.registry import MODELS

@MODELS.register_module()
class DSPHeadImage(BaseModule):
    def __init__(self,
                 n_classes,
                 in_channels,
                 out_channels,
                 n_reg_outs,
                 assigner,
                 prune_threshold=0,
                 bbox_loss=dict(type='IoULoss', reduction='none'),
                 cls_loss=dict(type='FocalLoss', reduction='none'),
                 keep_loss=dict(type='FocalLoss', reduction='mean', use_sigmoid=True),
                 train_cfg=None,
                 test_cfg=None):
        super(DSPHeadImage, self).__init__()
        self.prune_threshold = prune_threshold
        self.assigner = MODELS.build(assigner)
        self.bbox_loss = MODELS.build(bbox_loss)
        self.loss_cls = MODELS.build(cls_loss)
        self.keep_loss = MODELS.build(keep_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)

    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        # Prediction heads
        self.bbox_conv = nn.Conv2d(out_channels, n_reg_outs, kernel_size=1)
        self.cls_conv = nn.Conv2d(out_channels, n_classes, kernel_size=1)

        # Keep/Prune heads for each level except the last
        self.keep_conv = nn.ModuleList([
            nn.Conv2d(out_channels, 1, kernel_size=1)
            for _ in range(len(in_channels) - 1)
        ])

        # Feature processing blocks
        for i in range(len(in_channels)):
            if i > 0:
                # Upsampling blocks
                self.__setattr__(
                    f'up_block_{i}',
                    self._make_up_block(in_channels[i], in_channels[i - 1]))

            # Lateral and output blocks
            self.__setattr__(
                f'lateral_block_{i}',
                self._make_block(in_channels[i], in_channels[i]))
            self.__setattr__(
                f'out_block_{i}',
                self._make_block(in_channels[i], out_channels))

    @staticmethod
    def _make_block(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    @staticmethod
    def _make_up_block(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def init_weights(self):
        nn.init.normal_(self.bbox_conv.weight, std=.01)
        nn.init.normal_(self.cls_conv.weight, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

        for conv in self.keep_conv:
            nn.init.normal_(conv.weight, std=.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not ('bbox_conv' in m._get_name() or
                                                 'cls_conv' in m._get_name() or
                                                 'keep_conv' in m._get_name()):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_single(self, x):
        # Get predictions for a single feature level
        bbox_pred = self.bbox_conv(x)
        cls_pred = self.cls_conv(x)

        # For pruning, use max class score
        prune_scores = cls_pred.max(dim=1, keepdim=True)[0]

        # Get spatial coordinates (H,W)
        h, w = x.shape[-2:]
        y_coords = torch.arange(h, device=x.device).float().view(h, 1).expand(h, w)
        x_coords = torch.arange(w, device=x.device).float().view(1, w).expand(h, w)
        points = torch.stack([x_coords, y_coords], dim=-1).view(-1, 2)

        return bbox_pred, cls_pred, points, prune_scores

    def forward(self, x_tuple, gt_bboxes, gt_labels, img_metas):
        """Process multi-level features with progressive pruning.

        Args:
            x_tuple (tuple[Tensor]): Features from multiple levels, ordered from
                high to low resolution (e.g., [x3, x2, x1, x0]).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image.
            gt_labels (list[Tensor]): Ground truth labels for each image.
            img_metas (list[dict]): Meta information for each image.
        """
        bbox_preds, cls_preds, points = [], [], []
        keep_preds, keep_gts = [], []

        # Start from the lowest resolution feature
        print(len(x_tuple))
        print(x_tuple[0].shape)
        x = x_tuple[-1]

        for i in range(len(x_tuple) - 1, -1, -1):
            if i < len(x_tuple) - 1:
                #print(x.shape)
                #print("step",i)
                # Prune based on previous level's predictions
                prune_mask = self._get_keep_mask(x, keep_preds[-1])
                keep_gts.append(prune_mask)

                # Upsample and merge with next higher resolution feature
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = F.interpolate(x, size=x_tuple[i].shape[-2:], mode='bilinear', align_corners=False)
                x = x + x_tuple[i]

                # Apply pruning
                x = self._prune_features(x, prune_mask)

            # Process current level
            x = self.__getattr__(f'lateral_block_{i}')(x)
            out = self.__getattr__(f'out_block_{i}')(x)

            # Get predictions
            bbox_pred, cls_pred, point, prune_scores = self._forward_single(out)
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)

            # Get keep predictions for next level (except last level)
            if i > 0:
                keep_pred = self.keep_conv[i - 1](out)
                keep_preds.append(keep_pred)

        # Reverse to get predictions from high to low resolution
        return bbox_preds[::-1], cls_preds[::-1], points[::-1], keep_preds[::-1], keep_gts[::-1]

    def _get_keep_mask(self, x, keep_pred):
        """Generate binary mask for feature pruning based on keep predictions."""
        with torch.no_grad():
            # Upsample keep predictions to match feature size
            keep_pred = F.interpolate(keep_pred, size=x.shape[-2:], mode='bilinear', align_corners=False)
            keep_mask = (keep_pred.sigmoid() > self.prune_threshold).float()
        return keep_mask#需要变成可学习的，和gt对齐

    def _prune_features_old(self, x, prune_mask):
        """Apply pruning mask to features."""
        print(x.shape)
        print(prune_mask.shape)
        print("????????????????????????")
        return x * prune_mask

    def _prune_features(self, x, prune_mask):
        if prune_mask.shape[-2:] != x.shape[-2:]:
            prune_mask = F.interpolate(prune_mask, size=x.shape[-2:], mode='nearest')
        return x * prune_mask
    def _bbox_to_loss(self, bbox):
        """Convert bbox predictions to loss format."""
        # For axis-aligned boxes: x1, y1, x2, y2
        return torch.stack([
            bbox[..., 0] - bbox[..., 2] / 2,
            bbox[..., 1] - bbox[..., 3] / 2,
            bbox[..., 0] + bbox[..., 2] / 2,
            bbox[..., 1] + bbox[..., 3] / 2
        ], dim=-1)

    def _bbox_pred_to_bbox(self, points, bbox_pred):
        """Convert predicted bbox offsets to absolute coordinates."""
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[..., 0] + bbox_pred[..., 0]
        y_center = points[..., 1] + bbox_pred[..., 1]
        w = bbox_pred[..., 2]
        h = bbox_pred[..., 3]

        return torch.stack([
            x_center,
            y_center,
            w,
            h
        ], dim=-1)

    def _loss_single(self, bbox_preds, cls_preds, points, gt_bboxes, gt_labels, img_meta):
        """Compute loss for a single image."""
        # Flatten predictions
        #print(bbox_preds)
        #print(len(bbox_preds))
        #print(bbox_preds[0].shape)
        print("gt_bboxes",len(gt_bboxes))
        print(gt_bboxes[0].shape)
        print(gt_bboxes)
        bbox_preds = [p if p.dim() == 4 else p.unsqueeze(0) for p in bbox_preds]
        cls_preds = [p if p.dim() == 4 else p.unsqueeze(0) for p in cls_preds]

        bbox_pred = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, 4) for p in bbox_preds])
        cls_pred = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, cls_preds[0].shape[1]) for p in cls_preds])
        print("gt_bboxes",len(gt_bboxes))
        print(gt_bboxes[0].shape)
        print(gt_bboxes)
        print("points:")
        print(points)
        print(points[0].shape)
        print("?????????????????????????????")
        points = torch.cat(points)
        print(points.shape)
        print("?????????????????????????????")

        # Assign gt boxes to points
        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)

        # Classification loss
        n_classes = cls_pred.shape[1]
        pos_mask = assigned_ids >= 0
        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)
        cls_loss = self.cls_loss(cls_pred, cls_targets)

        # Bbox loss
        if pos_mask.sum() > 0:
            pos_bbox_pred = bbox_pred[pos_mask]
            pos_points = points[pos_mask]
            pos_bbox_targets = gt_bboxes[assigned_ids][pos_mask]

            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(self._bbox_pred_to_bbox(pos_points, pos_bbox_pred)),
                self._bbox_to_loss(pos_bbox_targets))
        else:
            bbox_loss = None

        return bbox_loss, cls_loss, pos_mask

    def _loss(self, bbox_preds, cls_preds, points, gt_bboxes, gt_labels, img_metas, keep_preds, keep_gts):
        """Compute total loss."""
        bbox_losses, cls_losses, pos_masks = [], [], []
        keep_losses = []

        # Keep loss
        for preds, gts in zip(keep_preds, keep_gts):
            for pred, gt in zip(preds, gts):
                #print("keep_loss:",pred.shape,gt.shape)
                keep_loss = self.keep_loss(pred, gt)
                keep_losses.append(keep_loss)
        keep_loss = torch.mean(torch.stack(keep_losses)) if keep_losses else 0

        #print("bbox_pred:")
        #print(len(bbox_preds))
        #print(bbox_preds[1].shape)
        #print(bbox_preds[0].shape)
        #print(points[0].shape)
        #print(points[1].shape)
        #print(points[2].shape)
        #print(points[3].shape)
        #print("!!!!!!!!!!!!!!!!!!")
        # Bbox and cls loss
        #print("len(img_metas)",len(img_metas))
        #print(img_metas[0])

        for i in range(len(img_metas)):
            bbox_loss, cls_loss, pos_mask = self._loss_single(
                bbox_preds=[p[i] for p in bbox_preds],
                cls_preds=[p[i] for p in cls_preds],
                points=[p[i] for p in points],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                img_meta=img_metas[i])

            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            pos_masks.append(pos_mask)

        return dict(
            bbox_loss=0.05*torch.mean(torch.stack(bbox_losses)) if bbox_losses else 0,
            cls_loss=torch.sum(torch.stack(cls_losses)) / torch.sum(torch.cat(pos_masks)),
            keep_loss=0.01 * keep_loss)

    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        #print("Input feature shape:", x[0].shape)
        bbox_preds, cls_preds, points, keep_preds, keep_gts = self(x, gt_bboxes, gt_labels, img_metas)
        #print(points[0].shape)
        #print(len(points))
        #print("2812818218281821")
        return self._loss(bbox_preds, cls_preds, points, gt_bboxes, gt_labels, img_metas, keep_preds, keep_gts)

    def _nms(self, bboxes, scores, img_meta):
        """Non-maximum suppression for 2D boxes."""
        n_classes = scores.shape[1]
        nms_bboxes, nms_scores, nms_labels = [], [], []

        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]

            keep = nms(class_bboxes, class_scores, self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[keep])
            nms_scores.append(class_scores[keep])
            nms_labels.append(
                bboxes.new_full(class_scores[keep].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, 4))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))

        return nms_bboxes, nms_scores, nms_labels

    def _get_bboxes_single(self, bbox_preds, cls_preds, points, img_meta):
        """Get final bboxes for a single image."""
        # Flatten predictions
        bbox_pred = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, 4) for p in bbox_preds])
        cls_pred = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, cls_preds[0].shape[1]) for p in cls_preds])
        points = torch.cat(points)
        scores = cls_pred.sigmoid()

        # Filter by score threshold
        if len(scores) > self.test_cfg.nms_pre > 0:
            max_scores, _ = scores.max(dim=1)
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_pred = bbox_pred[ids]
            scores = scores[ids]
            points = points[ids]

        # Convert to absolute coordinates
        boxes = self._bbox_pred_to_bbox(points, bbox_pred)
        boxes = self._bbox_to_loss(boxes)  # Convert to x1,y1,x2,y2 format

        # Apply NMS
        boxes, scores, labels = self._nms(boxes, scores, img_meta)
        return boxes, scores, labels

    def forward_test(self, x, img_metas):
        """Test forward with pruning."""
        bbox_preds, cls_preds, points = [], [], []
        keep_pred = None

        # Start from lowest resolution
        x = x[-1]

        for i in range(len(x) - 1, -1, -1):
            if i < len(x) - 1:
                # Prune based on previous level's predictions
                if keep_pred is not None:
                    prune_mask = self._get_keep_mask(x, keep_pred)
                    x = self._prune_features(x, prune_mask)

                # Upsample and merge
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = F.interpolate(x, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
                x = x + x[i]

            # Process current level
            x = self.__getattr__(f'lateral_block_{i}')(x)
            out = self.__getattr__(f'out_block_{i}')(x)

            # Get predictions
            bbox_pred, cls_pred, point, _ = self._forward_single(out)
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)

            # Get keep predictions for next level (except last level)
            if i > 0:
                keep_pred = self.keep_conv[i - 1](out)

        # Get final detections
        return self._get_bboxes_single(
            bbox_preds[::-1], cls_preds[::-1], points[::-1], img_metas[0])


@MODELS.register_module()
class DSPAssignerImage:
    def __init__(self, top_pts_threshold):
        # top_pts_threshold: per box
        self.top_pts_threshold = top_pts_threshold

    @torch.no_grad()
    def assign(self, points, gt_bboxes, gt_labels, img_meta):
        """Assign ground truth boxes to feature points.

        Args:
            points (list[Tensor]): Points from multiple feature levels,
                each of shape (N, 2) for (x,y) coordinates.
            gt_bboxes (Tensor): Ground truth boxes of shape (M, 4) in
                (x1,y1,x2,y2) format.
            gt_labels (Tensor): Ground truth labels of shape (M,).
            img_meta (dict): Image meta information.

        Returns:
            Tensor: Assigned box indices for each point (-1 for background).
        """
        float_max = points[0].new_tensor(1e8)
        levels = torch.cat([points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
                            for i in range(len(points))])
        #points = torch.cat(points)
        n_points = len(points)
        n_boxes = len(gt_bboxes)

        if n_boxes == 0:
            return gt_labels.new_full((n_points,), -1)

        # Convert boxes to (cx, cy, w, h) format
        boxes = torch.stack([
            (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2,  # cx
            (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2,  # cy
            gt_bboxes[:, 2] - gt_bboxes[:, 0],  # w
            gt_bboxes[:, 3] - gt_bboxes[:, 1]  # h
        ], dim=-1)

        # 打印调试信息
        #print("points:", len(points), [p.shape for p in points])  # 输出: 4 [(65536,2), (16384,2), (4096,2), (1024,2)]
        #print("boxes:", boxes.shape)  # 输出: (1, 4)

        # 拼接所有点成一个张量 (n_points, 2)
        points = torch.cat(points)  # shape: (65536+16384+4096+1024, 2) = (87040, 2)
        n_points = points.shape[0]

        # 确保 boxes 是 (n_boxes, 4)
        if boxes.dim() == 2 and boxes.shape[0] == 1:
            boxes = boxes.squeeze(0)  # 变成 (4,)
        boxes = boxes.unsqueeze(0)  # 变成 (1, 4)
        #print(boxes.shape)
        #print("!!!!!!!!!!!!!!")
        n_boxes = boxes.shape[1]

        # 将 boxes 扩展到 (n_points, n_boxes, 4)
        boxes = boxes.to(points.device).expand(n_points, n_boxes, 4)  # shape: (87040, 1, 4)

        # 将 points 扩展到 (n_points, n_boxes, 2)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 2)  # shape: (87040, 1, 2)

        # Condition 1: Keep topk locations per box by center distance
        centers = boxes[..., :2]
        center_distances = torch.sum(torch.pow(centers - points, 2), dim=-1)

        # Condition 2: Only consider points within expanded boxes
        # Expand boxes by a factor based on feature level
        expanded_boxes = boxes.clone()
        level_factors = 2.0 ** (levels.float().unsqueeze(1).expand(n_points, n_boxes))
        expanded_boxes[..., 2:] = boxes[..., 2:] * level_factors.unsqueeze(-1)  # Expand w,h

        # Check if points are inside expanded boxes
        dx_min = points[..., 0] - (expanded_boxes[..., 0] - expanded_boxes[..., 2] / 2)
        dx_max = (expanded_boxes[..., 0] + expanded_boxes[..., 2] / 2) - points[..., 0]
        dy_min = points[..., 1] - (expanded_boxes[..., 1] - expanded_boxes[..., 3] / 2)
        dy_max = (expanded_boxes[..., 1] + expanded_boxes[..., 3] / 2) - points[..., 1]

        inside_box_mask = torch.stack([dx_min, dx_max, dy_min, dy_max], dim=-1).min(dim=-1)[0] > 0
        center_distances = torch.where(inside_box_mask, center_distances, float_max)

        # Condition 3: For each box, keep topk closest points
        topk_distances = torch.topk(
            center_distances,
            min(self.top_pts_threshold + 1, n_points),
            largest=False, dim=0).values[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)

        # Condition 4: For each point, assign to closest box among topk candidates
        center_distances = torch.where(topk_condition, center_distances, float_max)
        min_values, min_ids = center_distances.min(dim=1)
        assigned_ids = torch.where(min_values < float_max, min_ids, -1)

        return assigned_ids