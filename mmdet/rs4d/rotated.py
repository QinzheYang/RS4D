# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.ops import box_iou_rotated
from torch import Tensor

#from .rotated_boxes import RotatedBoxes

# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List

from mmengine.dataset import BaseDataset

from mmdet.registry import DATASETS

# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Tuple, TypeVar, Union

import cv2
import numpy as np
import torch
from mmdet.structures.bbox import BaseBoxes, register_box
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from torch import BoolTensor, Tensor

T = TypeVar('T')
DeviceType = Union[str, torch.device]
MaskType = Union[BitmapMasks, PolygonMasks]
# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Tuple, TypeVar, Union

import cv2
from mmdet.structures.bbox import BaseBoxes, register_box
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from torch import BoolTensor, Tensor
from mmdet.structures.bbox import (HorizontalBoxes, bbox_overlaps,
                                   get_box_tensor)
from mmcv.transforms.utils import cache_randomness
import copy
from typing import List, Optional, Tuple
from mmcv.cnn import ConvModule, Scale, is_norm
from mmdet.models import inverse_sigmoid
from mmdet.models.dense_heads import RTMDetHead
from mmdet.models.task_modules import anchor_inside_flags
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, sigmoid_geometric_mean,
                                unmap)
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, cat_boxes, distance2bbox
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmengine.structures import InstanceData
from torch import Tensor, nn
from mmengine.registry import TASK_UTILS
# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder
from mmdet.registry import MODELS

try:
    from mmcv.ops import diff_iou_rotated_2d
except:  # noqa: E722
    diff_iou_rotated_2d = None

# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS

import copy
import os
import os.path as osp
import re
import tempfile
import zipfile
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from mmcv.ops import nms_quadri, nms_rotated
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger
from mmdet.registry import METRICS

import cv2
from mmdet.structures.bbox import HorizontalBoxes, register_box_converter
from torch import Tensor
from multiprocessing import get_context

import numpy as np
import torch
from mmcv.ops import box_iou_quadri, box_iou_rotated
from mmdet.evaluation.functional import average_precision
from mmengine.logging import print_log
from terminaltables import AsciiTable
def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 box_type='rbox',
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 5). Defaults to None
        iou_thr (float): IoU threshold to be considered as matched.
            Defaults to 0.5.
        box_type (str): Box type. If the QuadriBoxes is used, you need to
            specify 'qbox'. Defaults to 'rbox'.
        area_ranges (list[tuple], optional): Range of bbox areas to be
            evaluated, in the format [(min1, max1), (min2, max2), ...].
            Defaults to None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """
    # an indicator of ignored gts
    det_bboxes = np.array(det_bboxes)
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0],
                  dtype=bool), np.ones(gt_bboxes_ignore.shape[0], dtype=bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        return tp, fp

    if box_type == 'rbox':
        ious = box_iou_rotated(
            torch.from_numpy(det_bboxes).float(),
            torch.from_numpy(gt_bboxes).float()).numpy()
    elif box_type == 'qbox':
        ious = box_iou_quadri(
            torch.from_numpy(det_bboxes).float(),
            torch.from_numpy(gt_bboxes).float()).numpy()
    else:
        raise NotImplementedError
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                if box_type == 'rbox':
                    bbox = det_bboxes[i, :5]
                    area = bbox[2] * bbox[3]
                elif box_type == 'qbox':
                    bbox = det_bboxes[i, :8]
                    pts = bbox.reshape(*bbox.shape[:-1], 4, 2)
                    roll_pts = torch.roll(pts, 1, dims=-2)
                    xyxy = torch.sum(
                        pts[..., 0] * roll_pts[..., 1] -
                        roll_pts[..., 0] * pts[..., 1],
                        dim=-1)
                    area = 0.5 * torch.abs(xyxy)
                else:
                    raise NotImplementedError
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp

def get_cls_results(det_results, annotations, class_id, box_type):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.
        box_type (str): Box type. If the QuadriBoxes is used, you need to
            specify 'qbox'. Defaults to 'rbox'.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]

    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        if len(ann['bboxes']) != 0:
            gt_inds = ann['labels'] == class_id
            cls_gts.append(ann['bboxes'][gt_inds, :])
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            if box_type == 'rbox':
                cls_gts.append(torch.zeros((0, 5), dtype=torch.float64))
                cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))
            elif box_type == 'qbox':
                cls_gts.append(torch.zeros((0, 8), dtype=torch.float64))
                cls_gts_ignore.append(torch.zeros((0, 8), dtype=torch.float64))
            else:
                raise NotImplementedError

    return cls_dets, cls_gts, cls_gts_ignore

def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str, optional): Dataset name or dataset classes.
        scale_ranges (list[tuple], optional): Range of scales to be evaluated.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details.
            Defaults to None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
def eval_rbbox_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   box_type='rbox',
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple], optional): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Defaults to None.
        iou_thr (float): IoU threshold to be considered as matched.
            Defaults to 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        box_type (str): Box type. If the QuadriBoxes is used, you need to
            specify 'qbox'. Defaults to 'rbox'.
        dataset (list[str] | str, optional): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Defaults to None.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details.
            Defaults to None.
        nproc (int): Processes used for computing TP and FP.
            Defaults to 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i, box_type)

        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [box_type for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                if box_type == 'rbox':
                    gt_areas = bbox[:, 2] * bbox[:, 3]
                elif box_type == 'qbox':
                    pts = bbox.reshape(*bbox.shape[:-1], 4, 2)
                    roll_pts = torch.roll(pts, 1, dims=-2)
                    xyxy = torch.sum(
                        pts[..., 0] * roll_pts[..., 1] -
                        roll_pts[..., 0] * pts[..., 1],
                        dim=-1)
                    gt_areas = 0.5 * torch.abs(xyxy)
                else:
                    raise NotImplementedError
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results
@register_box('qbox')
class QuadriBoxes(BaseBoxes):
    """The quadrilateral box class.

    The ``box_dim`` of ``QuadriBoxes`` is 8, which means the length of the
    last dimension of the input should be 8. Each row of data means (x1, y1,
    x2, y2, x3, y3, x4, y4) which are the coordinates of 4 vertices of the box.
    The box must be convex. The order of 4 vertices can be both CW and CCW.

    ``QuadriBoxes`` usually works as the raw data loaded from dataset like
    DOTA, DIOR, etc.

    Args:
        boxes (Tensor or np.ndarray or Sequence): The box data with
            shape (..., 8).
        dtype (torch.dtype, Optional): data type of boxes. Defaults to None.
        device (str or torch.device, Optional): device of boxes.
            Default to None.
        clone (bool): Whether clone ``boxes`` or not. Defaults to True.
    """

    box_dim = 8

    @property
    def vertices(self) -> Tensor:
        """Return a tensor representing the vertices of boxes.

        If boxes have shape of (m, 8), vertices have shape of (m, 4, 2)
        """
        boxes = self.tensor
        return boxes.reshape(*boxes.shape[:-1], 4, 2)

    @property
    def centers(self) -> Tensor:
        """Return a tensor representing the centers of boxes.

        If boxes have shape of (m, 8), centers have shape of (m, 2).
        """
        boxes = self.tensor
        boxes = boxes.reshape(*boxes.shape[:-1], 4, 2)
        return boxes.mean(dim=-2)

    @property
    def areas(self) -> Tensor:
        """Return a tensor representing the areas of boxes.

        If boxes have shape of (m, 8), areas have shape of (m, ).
        """
        boxes = self.tensor
        pts = boxes.reshape(*boxes.shape[:-1], 4, 2)
        roll_pts = torch.roll(pts, 1, dims=-2)
        xyxy = torch.sum(
            pts[..., 0] * roll_pts[..., 1] - roll_pts[..., 0] * pts[..., 1],
            dim=-1)
        areas = 0.5 * torch.abs(xyxy)
        return areas

    @property
    def widths(self) -> Tensor:
        """Return a tensor representing the widths of boxes.

        If boxes have shape of (m, 8), widths have shape of (m, ).

        notes:
            Quadrilateral boxes don't have the width concept. Use
            ``sqrt(areas)`` to replace the width.
        """
        warnings.warn("Quadrilateral boxes don't have the width concept. "
                      'We use ``sqrt(areas)`` to replace the width.')
        return torch.sqrt(self.areas)

    @property
    def heights(self) -> Tensor:
        """Return a tensor representing the heights of boxes.

        If boxes have shape of (m, 8), heights have shape of (m, ).

        notes:
            Quadrilateral boxes don't have the height concept. Use
            ``sqrt(areas)`` to replace the heights.
        """
        warnings.warn("Quadrilateral boxes don't have the height concept. "
                      'We use ``sqrt(areas)`` to replace the width.')
        return torch.sqrt(self.areas)

    def flip_(self,
              img_shape: Tuple[int, int],
              direction: str = 'horizontal') -> None:
        """Flip boxes horizontally or vertically in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        flipped = self.tensor
        if direction == 'horizontal':
            flipped[..., 0::2] = img_shape[1] - flipped[..., 0::2]
        elif direction == 'vertical':
            flipped[..., 1::2] = img_shape[0] - flipped[..., 1::2]
        else:
            flipped[..., 0::2] = img_shape[1] - flipped[..., 0::2]
            flipped[..., 1::2] = img_shape[0] - flipped[..., 1::2]

    def translate_(self, distances: Tuple[float, float]) -> None:
        """Translate boxes in-place.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        boxes = self.tensor
        assert len(distances) == 2
        self.tensor = boxes + boxes.new_tensor(distances).repeat(4)

    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Clip boxes according to the image shape in-place.

        In ``QuadriBoxes``, ``clip`` function does nothing about the original
        data, because it's very tricky to handle rotate boxes corssing the
        image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            T: Cliped boxes with the same shape as the original boxes.
        """
        warnings.warn('The `clip` function does nothing in `QuadriBoxes`.')

    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        """Rotate all boxes in-place.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle represented in degrees. Positive
                values mean clockwise rotation.
        """
        boxes = self.tensor
        rotation_matrix = boxes.new_tensor(
            cv2.getRotationMatrix2D(center, -angle, 1))

        corners = boxes.reshape(*boxes.shape[:-1], 4, 2)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(rotation_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        self.tensor = corners.reshape(*corners.shape[:-2], 8)

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Geometric transformat boxes in-place.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
        """
        boxes = self.tensor
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = boxes.new_tensor(homography_matrix)
        corners = boxes.reshape(*boxes.shape[:-1], 4, 2)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(homography_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        # Convert to homogeneous coordinates by normalization
        corners = corners[..., :2] / corners[..., 2:3]
        self.tensor = corners.reshape(*corners.shape[:-2], 8)

    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        """Rescale boxes w.r.t. rescale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
        """
        boxes = self.tensor
        assert len(scale_factor) == 2
        scale_factor = boxes.new_tensor(scale_factor).repeat(4)
        self.tensor = boxes * scale_factor

    def resize_(self, scale_factor: Tuple[float, float]) -> None:
        """Resize the box width and height w.r.t scale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.
        """
        boxes = self.tensor
        assert len(scale_factor) == 2
        assert scale_factor[0] == scale_factor[1], \
            'To protect the shape of QuadriBoxes not changes'
        scale_factor = boxes.new_tensor(scale_factor)

        boxes = boxes.reshape(*boxes.shape[:-1], 4, 2)
        centers = boxes.mean(dim=-2)[..., None, :]
        boxes = (boxes - centers) * scale_factor + centers
        self.tensor = boxes.reshape(*boxes.shape[:-2], 8)

    def is_inside(self,
                  img_shape: Tuple[int, int],
                  all_inside: bool = False,
                  allowed_border: int = 0) -> BoolTensor:
        """Find boxes inside the image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            all_inside (bool): Whether the boxes are all inside the image or
                part inside the image. Defaults to False.
            allowed_border (int): Boxes that extend beyond the image shape
                boundary by more than ``allowed_border`` are considered
                "outside" Defaults to 0.

        Returns:
            BoolTensor: A BoolTensor indicating whether the box is inside
            the image. Assuming the original boxes have shape (m, n, 8),
            the output has shape (m, n).
        """
        img_h, img_w = img_shape
        boxes = self.tensor
        boxes = boxes.reshape(*boxes.shape[:-1], 4, 2)
        centers = boxes.mean(dim=-2)
        return (centers[..., 0] <= img_w + allowed_border) & \
               (centers[..., 1] <= img_h + allowed_border) & \
               (centers[..., 0] >= -allowed_border) & \
               (centers[..., 1] >= -allowed_border)

    def find_inside_points(self,
                           points: Tensor,
                           is_aligned: bool = False,
                           eps: float = 0.01) -> BoolTensor:
        """Find inside box points. Boxes dimension must be 2.
        Args:
            points (Tensor): Points coordinates. Has shape of (m, 2).
            is_aligned (bool): Whether ``points`` has been aligned with boxes
                or not. If True, the length of boxes and ``points`` should be
                the same. Defaults to False.
            eps (float): Make sure the points are inside not on the boundary.
                Defaults to 0.01.

        Returns:
            BoolTensor: A BoolTensor indicating whether a point is inside
            boxes. Assuming the boxes has shape of (n, 8), if ``is_aligned``
            is False. The index has shape of (m, n). If ``is_aligned`` is
            True, m should be equal to n and the index has shape of (m, ).
        """
        boxes = self.tensor
        assert boxes.dim() == 2, 'boxes dimension must be 2.'

        corners = boxes.reshape(-1, 4, 2)
        corners_next = torch.roll(corners, -1, dims=1)
        x1, y1 = corners.unbind(dim=2)
        x2, y2 = corners_next.unbind(dim=2)
        pt_x, pt_y = points.split([1, 1], dim=1)

        if not is_aligned:
            pt_x = pt_x[:, None, :]
            pt_y = pt_y[:, None, :]
            x1 = x1[None, :, :]
            y1 = y1[None, :, :]
            x2 = x2[None, :, :]
            y2 = y2[None, :, :]
        else:
            assert boxes.size(0) == points.size(0)

        values = (x1 - pt_x) * (y2 - pt_y) - (y1 - pt_y) * (x2 - pt_x)
        return (values >= eps).all(dim=-1) | (values <= -eps).all(dim=-1)

    @staticmethod
    def overlaps(boxes1: BaseBoxes,
                 boxes2: BaseBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False,
                 eps: float = 1e-6) -> Tensor:
        """Calculate overlap between two set of boxes with their modes
        converted to ``QuadriBoxes``.

        Args:
            boxes1 (:obj:`BaseBoxes`): BaseBoxes with shape of (m, box_dim)
                or empty.
            boxes2 (:obj:`BaseBoxes`): BaseBoxes with shape of (n, box_dim)
                or empty.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground). Defaults to "iou".
            is_aligned (bool): If True, then m and n must be equal. Defaults
                to False.
            eps (float): A value added to the denominator for numerical
                stability. Defaults to 1e-6.

        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
        """
        raise NotImplementedError

    def from_instance_masks(masks: MaskType) -> 'QuadriBoxes':
        """Create boxes from instance masks.

        Args:
            masks (:obj:`BitmapMasks` or :obj:`PolygonMasks`): BitmapMasks or
                PolygonMasks instance with length of n.

        Returns:
            :obj:`QuadriBoxes`: Converted boxes with shape of (n, 8).
        """
        num_masks = len(masks)
        if num_masks == 0:
            return QuadriBoxes([], dtype=torch.float32)

        boxes = []
        if isinstance(masks, PolygonMasks):
            for idx, poly_per_obj in enumerate(masks.masks):
                pts_per_obj = []
                for p in poly_per_obj:
                    pts_per_obj.append(
                        np.array(p, dtype=np.float32).reshape(-1, 2))
                pts_per_obj = np.concatenate(pts_per_obj, axis=0)
                rect = cv2.minAreaRect(pts_per_obj)
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = cv2.boxPoints(rect)
                boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
        else:
            masks = masks.to_ndarray()
            for idx in range(num_masks):
                coor_y, coor_x = np.nonzero(masks[idx])
                points = np.stack([coor_x, coor_y], axis=-1).astype(np.float32)
                rect = cv2.minAreaRect(points)
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = cv2.boxPoints(rect)
                boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
        return QuadriBoxes(boxes)
@register_box('rbox')
class RotatedBoxes(BaseBoxes):
    """The rotated box class used in MMRotate by default.

    The ``box_dim`` of ``RotatedBoxes`` is 5, which means the length of the
    last dimension of the input should be 5. Each row of data means
    (x, y, w, h, t), where 'x' and 'y' are the coordinates of the box center,
    'w' and 'h' are the length of box sides, 't' is the box angle represented
    in radian. A rotated box can be regarded as rotating the horizontal box
    (x, y, w, h) w.r.t its center by 't' radian CW.

    Args:
        data (Tensor or np.ndarray or Sequence): The box data with shape
            (..., 5).
        dtype (torch.dtype, Optional): data type of boxes. Defaults to None.
        device (str or torch.device, Optional): device of boxes.
            Default to None.
        clone (bool): Whether clone ``boxes`` or not. Defaults to True.
    """

    box_dim = 5

    def regularize_boxes(self,
                         pattern: Optional[str] = None,
                         width_longer: bool = True,
                         start_angle: float = -90) -> Tensor:
        """Regularize rotated boxes.

        Due to the angle periodicity, one rotated box can be represented in
        many different (x, y, w, h, t). To make each rotated box unique,
        ``regularize_boxes`` will take the remainder of the angle divided by
        180 degrees.

        However, after taking the remainder of the angle, there are still two
        representations for one rotate box. For example, (0, 0, 4, 5, 0.5) and
        (0, 0, 5, 4, 0.5 + pi/2) are the same areas in the image. To solve the
        problem, the code will swap edges w.r.t ``width_longer``:

        - width_longer=True: Make sure the width is longer than the height. If
          not, swap the width and height. The angle ranges in [start_angle,
          start_angle + 180). For the above example, the rotated box will be
          represented as (0, 0, 5, 4, 0.5 + pi/2).
        - width_longer=False: Make sure the angle is lower than
          start_angle+pi/2. If not, swap the width and height. The angle
          ranges in [start_angle, start_angle + 90). For the above example,
          the rotated box will be represented as (0, 0, 4, 5, 0.5).

        For convenience, three commonly used patterns are preset in
        ``regualrize_boxes``:

        - 'oc': OpenCV Definition. Has the same box representation as
          ``cv2.minAreaRect`` the angle ranges in [-90, 0). Equal to set
          width_longer=False and start_angle=-90.
        - 'le90': Long Edge Definition (90). the angle ranges in [-90, 90).
          The width is always longer than the height. Equal to set
          width_longer=True and start_angle=-90.
        - 'le135': Long Edge Definition (135). the angle ranges in [-45, 135).
          The width is always longer than the height. Equal to set
          width_longer=True and start_angle=-45.

        Args:
            pattern (str, Optional): Regularization pattern. Can only be 'oc',
                'le90', or 'le135'. Defaults to None.
            width_longer (bool): Whether to make sure width is larger than
                height. Defaults to True.
            start_angle (float): The starting angle of the box angle
                represented in degrees. Defaults to -90.

        Returns:
            Tensor: Regularized box tensor.
        """
        boxes = self.tensor
        if pattern is not None:
            if pattern == 'oc':
                width_longer, start_angle = False, -90
            elif pattern == 'le90':
                width_longer, start_angle = True, -90
            elif pattern == 'le135':
                width_longer, start_angle = True, -45
            else:
                raise ValueError("pattern only can be 'oc', 'le90', and"
                                 f"'le135', but get {pattern}.")
        start_angle = start_angle / 180 * np.pi

        x, y, w, h, t = boxes.unbind(dim=-1)
        if width_longer:
            # swap edge and angle if h >= w
            w_ = torch.where(w > h, w, h)
            h_ = torch.where(w > h, h, w)
            t = torch.where(w > h, t, t + np.pi / 2)
            t = ((t - start_angle) % np.pi) + start_angle
        else:
            # swap edge and angle if angle > pi/2
            t = ((t - start_angle) % np.pi)
            w_ = torch.where(t < np.pi / 2, w, h)
            h_ = torch.where(t < np.pi / 2, h, w)
            t = torch.where(t < np.pi / 2, t, t - np.pi / 2) + start_angle
        self.tensor = torch.stack([x, y, w_, h_, t], dim=-1)
        return self.tensor

    @property
    def centers(self) -> Tensor:
        """Return a tensor representing the centers of boxes.

        If boxes have shape of (m, 8), centers have shape of (m, 2).
        """
        return self.tensor[..., :2]

    @property
    def areas(self) -> Tensor:
        """Return a tensor representing the areas of boxes.

        If boxes have shape of (m, 8), areas have shape of (m, ).
        """
        return self.tensor[..., 2] * self.tensor[..., 3]

    @property
    def widths(self) -> Tensor:
        """Return a tensor representing the widths of boxes.

        If boxes have shape of (m, 8), widths have shape of (m, ).
        """
        return self.tensor[..., 2]

    @property
    def heights(self) -> Tensor:
        """Return a tensor representing the heights of boxes.

        If boxes have shape of (m, 8), heights have shape of (m, ).
        """
        return self.tensor[..., 3]

    def flip_(self,
              img_shape: Tuple[int, int],
              direction: str = 'horizontal') -> None:
        """Flip boxes horizontally or vertically in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        flipped = self.tensor
        if direction == 'horizontal':
            flipped[..., 0] = img_shape[1] - flipped[..., 0]
            flipped[..., 4] = -flipped[..., 4]
        elif direction == 'vertical':
            flipped[..., 1] = img_shape[0] - flipped[..., 1]
            flipped[..., 4] = -flipped[..., 4]
        else:
            flipped[..., 0] = img_shape[1] - flipped[..., 0]
            flipped[..., 1] = img_shape[0] - flipped[..., 1]

    def translate_(self, distances: Tuple[float, float]) -> None:
        """Translate boxes in-place.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        boxes = self.tensor
        assert len(distances) == 2
        boxes[..., :2] = boxes[..., :2] + boxes.new_tensor(distances)

    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Clip boxes according to the image shape in-place.

        In ``RotatedBoxes``, ``clip`` function does nothing about the original
        data, because it's very tricky to handle rotate boxes corssing the
        image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
        """
        warnings.warn('The `clip` function does nothing in `RotatedBoxes`.')

    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        """Rotate all boxes in-place.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle represented in degrees. Positive
                values mean clockwise rotation.
        """
        boxes = self.tensor
        rotation_matrix = boxes.new_tensor(
            cv2.getRotationMatrix2D(center, -angle, 1))

        centers, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        t = t + angle / 180 * np.pi
        centers = torch.cat(
            [centers, centers.new_ones(*centers.shape[:-1], 1)], dim=-1)
        centers_T = torch.transpose(centers, -1, -2)
        centers_T = torch.matmul(rotation_matrix, centers_T)
        centers = torch.transpose(centers_T, -1, -2)
        self.tensor = torch.cat([centers, wh, t], dim=-1)

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Geometric transformat boxes in-place.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
        """
        boxes = self.tensor
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = boxes.new_tensor(homography_matrix)
        corners = self.rbox2corner(boxes)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(homography_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        # Convert to homogeneous coordinates by normalization
        corners = corners[..., :2] / corners[..., 2:3]
        self.tensor = self.corner2rbox(corners)

    @staticmethod
    def rbox2corner(boxes: Tensor) -> Tensor:
        """Convert rotated box (x, y, w, h, t) to corners ((x1, y1), (x2, y1),
        (x1, y2), (x2, y2)).

        Args:
            boxes (Tensor): Rotated box tensor with shape of (..., 5).

        Returns:
            Tensor: Corner tensor with shape of (..., 4, 2).
        """
        ctr, w, h, theta = torch.split(boxes, (2, 1, 1, 1), dim=-1)
        cos_value, sin_value = torch.cos(theta), torch.sin(theta)
        vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
        vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        return torch.stack([pt1, pt2, pt3, pt4], dim=-2)

    @staticmethod
    def corner2rbox(corners: Tensor) -> Tensor:
        """Convert corners ((x1, y1), (x2, y1), (x1, y2), (x2, y2)) to rotated
        box (x, y, w, h, t).

        Args:
            corners (Tensor): Corner tensor with shape of (..., 4, 2).

        Returns:
            Tensor: Rotated box tensor with shape of (..., 5).
        """
        original_shape = corners.shape[:-2]
        points = corners.cpu().numpy().reshape(-1, 4, 2)
        rboxes = []
        for pts in points:
            (x, y), (w, h), angle = cv2.minAreaRect(pts)
            rboxes.append([x, y, w, h, angle / 180 * np.pi])
        rboxes = corners.new_tensor(rboxes)
        return rboxes.reshape(*original_shape, 5)

    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        """Rescale boxes w.r.t. rescale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
        """
        boxes = self.tensor
        assert len(scale_factor) == 2
        scale_x, scale_y = scale_factor
        ctrs, w, h, t = torch.split(boxes, [2, 1, 1, 1], dim=-1)
        cos_value, sin_value = torch.cos(t), torch.sin(t)

        # Refer to https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/rotated_boxes.py # noqa
        # rescale centers
        ctrs = ctrs * ctrs.new_tensor([scale_x, scale_y])
        # rescale width and height
        w = w * torch.sqrt((scale_x * cos_value)**2 + (scale_y * sin_value)**2)
        h = h * torch.sqrt((scale_x * sin_value)**2 + (scale_y * cos_value)**2)
        # recalculate theta
        t = torch.atan2(scale_x * sin_value, scale_y * cos_value)
        self.tensor = torch.cat([ctrs, w, h, t], dim=-1)

    def resize_(self, scale_factor: Tuple[float, float]) -> None:
        """Resize the box width and height w.r.t scale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.
        """
        boxes = self.tensor
        assert len(scale_factor) == 2
        ctrs, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        scale_factor = boxes.new_tensor(scale_factor)
        wh = wh * scale_factor
        self.tensor = torch.cat([ctrs, wh, t], dim=-1)

    def is_inside(self,
                  img_shape: Tuple[int, int],
                  all_inside: bool = False,
                  allowed_border: int = 0) -> BoolTensor:
        """Find boxes inside the image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            all_inside (bool): Whether the boxes are all inside the image or
                part inside the image. Defaults to False.
            allowed_border (int): Boxes that extend beyond the image shape
                boundary by more than ``allowed_border`` are considered
                "outside" Defaults to 0.

        Returns:
            BoolTensor: A BoolTensor indicating whether the box is inside
            the image. Assuming the original boxes have shape (m, n, 5),
            the output has shape (m, n).
        """
        img_h, img_w = img_shape
        boxes = self.tensor
        return (boxes[..., 0] <= img_w + allowed_border) & \
               (boxes[..., 1] <= img_h + allowed_border) & \
               (boxes[..., 0] >= -allowed_border) & \
               (boxes[..., 1] >= -allowed_border)

    def find_inside_points(self,
                           points: Tensor,
                           is_aligned: bool = False,
                           eps: float = 0.01) -> BoolTensor:
        """Find inside box points. Boxes dimension must be 2.
        Args:
            points (Tensor): Points coordinates. Has shape of (m, 2).
            is_aligned (bool): Whether ``points`` has been aligned with boxes
                or not. If True, the length of boxes and ``points`` should be
                the same. Defaults to False.
            eps (float): Make sure the points are inside not on the boundary.
                Defaults to 0.01.

        Returns:
            BoolTensor: A BoolTensor indicating whether the box is inside the
            image. Assuming the boxes has shape of (n, 5), if ``is_aligned``
            is False. The index has shape of (m, n). If ``is_aligned`` is True,
            m should be equal to n and the index has shape of (m, ).
        """
        boxes = self.tensor
        assert boxes.dim() == 2, 'boxes dimension must be 2.'

        if not is_aligned:
            boxes = boxes[None, :, :]
            points = points[:, None, :]
        else:
            assert boxes.size(0) == points.size(0)

        ctrs, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        cos_value, sin_value = torch.cos(t), torch.sin(t)
        matrix = torch.cat([cos_value, sin_value, -sin_value, cos_value],
                           dim=-1).reshape(*boxes.shape[:-1], 2, 2)

        offset = points - ctrs
        offset = torch.matmul(matrix, offset[..., None])
        offset = offset.squeeze(-1)
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        w, h = wh[..., 0], wh[..., 1]
        return (offset_x <= w / 2 - eps) & (offset_x >= - w / 2 + eps) & \
            (offset_y <= h / 2 - eps) & (offset_y >= - h / 2 + eps)

    @staticmethod
    def overlaps(boxes1: BaseBoxes,
                 boxes2: BaseBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False,
                 eps: float = 1e-6) -> Tensor:
        """Calculate overlap between two set of boxes with their types
        converted to ``RotatedBoxes``.

        Args:
            boxes1 (:obj:`BaseBoxes`): BaseBoxes with shape of (m, box_dim)
                or empty.
            boxes2 (:obj:`BaseBoxes`): BaseBoxes with shape of (n, box_dim)
                or empty.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground). Defaults to "iou".
            is_aligned (bool): If True, then m and n must be equal. Defaults
                to False.
            eps (float): A value added to the denominator for numerical
                stability. Defaults to 1e-6.

        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
        """
        from mmdet.rotated_rtmdet.bbox_overlaps import rbbox_overlaps
        boxes1 = boxes1.convert_to('rbox')
        boxes2 = boxes2.convert_to('rbox')
        return rbbox_overlaps(
            boxes1.tensor,
            boxes2.tensor,
            mode=mode,
            is_aligned=is_aligned,
            eps=eps)

    @staticmethod
    def from_instance_masks(masks: MaskType) -> 'RotatedBoxes':
        """Create boxes from instance masks.

        Args:
            masks (:obj:`BitmapMasks` or :obj:`PolygonMasks`): BitmapMasks or
                PolygonMasks instance with length of n.

        Returns:
            :obj:`RotatedBoxes`: Converted boxes with shape of (n, 5).
        """
        num_masks = len(masks)
        if num_masks == 0:
            return RotatedBoxes([], dtype=torch.float32)

        boxes = []
        if isinstance(masks, BitmapMasks):
            for idx in range(num_masks):
                mask = masks.masks[idx]
                points = np.stack(np.nonzero(mask), axis=-1).astype(np.float32)
                (x, y), (w, h), angle = cv2.minAreaRect(points)
                boxes.append([x, y, w, h, angle / 180 * np.pi])
        elif isinstance(masks, PolygonMasks):
            for idx, poly_per_obj in enumerate(masks.masks):
                pts_per_obj = []
                for p in poly_per_obj:
                    pts_per_obj.append(
                        np.array(p, dtype=np.float32).reshape(-1, 2))
                pts_per_obj = np.concatenate(pts_per_obj, axis=0)
                (x, y), (w, h), angle = cv2.minAreaRect(pts_per_obj)
                boxes.append([x, y, w, h, angle / 180 * np.pi])
        else:
            raise TypeError(
                '`masks` must be `BitmapMasks`  or `PolygonMasks`, '
                f'but got {type(masks)}.')
        return RotatedBoxes(boxes)

@register_box_converter(RotatedBoxes, QuadriBoxes)
def rbox2qbox(boxes: Tensor) -> Tensor:
    """Convert rotated boxes to quadrilateral boxes.

    Args:
        boxes (Tensor): Rotated box tensor with shape of (..., 5).

    Returns:
        Tensor: Quadrilateral box tensor with shape of (..., 8).
    """
    ctr, w, h, theta = torch.split(boxes, (2, 1, 1, 1), dim=-1)
    cos_value, sin_value = torch.cos(theta), torch.sin(theta)
    vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
    vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return torch.cat([pt1, pt2, pt3, pt4], dim=-1)

@register_box_converter(QuadriBoxes, RotatedBoxes)
def qbox2rbox(boxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to rotated boxes.

    Args:
        boxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    # TODO support tensor-based minAreaRect later
    original_shape = boxes.shape[:-1]
    points = boxes.cpu().numpy().reshape(-1, 4, 2)
    rboxes = []
    for pts in points:
        (x, y), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([x, y, w, h, angle / 180 * np.pi])
    rboxes = boxes.new_tensor(rboxes)
    return rboxes.view(*original_shape, 5)

@register_box_converter(HorizontalBoxes, RotatedBoxes)
def hbox2rbox(boxes: Tensor) -> Tensor:
    """Convert horizontal boxes to rotated boxes.

    Args:
        boxes (Tensor): horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    wh = boxes[..., 2:] - boxes[..., :2]
    ctrs = (boxes[..., 2:] + boxes[..., :2]) / 2
    theta = boxes.new_zeros((*boxes.shape[:-1], 1))
    return torch.cat([ctrs, wh, theta], dim=-1)

@register_box_converter(HorizontalBoxes, QuadriBoxes)
def hbox2qbox(boxes: Tensor) -> Tensor:
    """Convert horizontal boxes to quadrilateral boxes.

    Args:
        boxes (Tensor): horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Quadrilateral box tensor with shape of (..., 8).
    """
    x1, y1, x2, y2 = torch.split(boxes, 1, dim=-1)
    return torch.cat([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1)

@register_box_converter(RotatedBoxes, HorizontalBoxes)
def rbox2hbox(boxes: Tensor) -> Tensor:
    """Convert rotated boxes to horizontal boxes.

    Args:
        boxes (Tensor): Rotated box tensor with shape of (..., 5).

    Returns:
        Tensor: Horizontal box tensor with shape of (..., 4).
    """
    ctrs, w, h, theta = torch.split(boxes, (2, 1, 1, 1), dim=-1)
    cos_value, sin_value = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * cos_value) + torch.abs(h / 2 * sin_value)
    y_bias = torch.abs(w / 2 * sin_value) + torch.abs(h / 2 * cos_value)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([ctrs - bias, ctrs + bias], dim=-1)

@register_box_converter(QuadriBoxes, HorizontalBoxes)
def qbox2hbox(boxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to horizontal boxes.

    Args:
        boxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

    Returns:
        Tensor: Horizontal box tensor with shape of (..., 4).
    """
    boxes = boxes.view(*boxes.shape[:-1], 4, 2)
    x1y1, _ = boxes.min(dim=-2)
    x2y2, _ = boxes.max(dim=-2)
    return torch.cat([x1y1, x2y2], dim=-1)

# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmdet.structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from mmdet.visualization import DetLocalVisualizer, jitter_color
from mmdet.visualization.palette import _get_adaptive_scales
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import VISUALIZERS
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import mmcv
from mmengine.utils import is_str


def get_palette(palette: Union[List[tuple], str, tuple],
                num_classes: int) -> List[Tuple[int]]:
    """Get palette from various inputs.

    Args:
        palette (list[tuple] | str | tuple): palette inputs.
        num_classes (int): the number of classes.
    Returns:
        list[tuple[int]]: A list of color tuples.
    """
    assert isinstance(num_classes, int)

    if isinstance(palette, list):
        dataset_palette = palette
    elif isinstance(palette, tuple):
        dataset_palette = [palette] * num_classes
    elif palette == 'random' or palette is None:
        state = np.random.get_state()
        # random color
        np.random.seed(42)
        palette = np.random.randint(0, 256, size=(num_classes, 3))
        np.random.set_state(state)
        dataset_palette = [tuple(c) for c in palette]
    elif palette == 'dota':
        from mmdet.datasets import DOTADataset
        dataset_palette = DOTADataset.METAINFO['palette']
    elif palette == 'sar':
        from mmdet.datasets import SARDataset
        dataset_palette = SARDataset.METAINFO['palette']
    elif palette == 'hrsc':
        from mmdet.datasets import HRSCDataset
        dataset_palette = HRSCDataset.METAINFO['palette']
    elif is_str(palette):
        dataset_palette = [mmcv.color_val(palette)[::-1]] * num_classes
    else:
        raise TypeError(f'Invalid type for palette: {type(palette)}')

    assert len(dataset_palette) >= num_classes, \
        'The length of palette should not be less than `num_classes`.'
    return dataset_palette

@VISUALIZERS.register_module()
class RotLocalVisualizer(DetLocalVisualizer):
    """MMRotate Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str, tuple(int), optional): Color of texts.
            The tuple of color should be in BGR order.
            Defaults to (200, 200, 200).
        mask_color (str, tuple(int), optional): Color of masks.
            The tuple of color should be in BGR order.
            Defaults to None.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of bboxes or mask.
            Defaults to 0.8.
    """

    def _draw_instances(self, image: np.ndarray, instances: ['InstanceData'],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.
        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances:
            bboxes = instances.bboxes
            labels = instances.labels

            max_label = int(max(labels) if len(labels) > 0 else 0)
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]

            if isinstance(bboxes, Tensor):
                if bboxes.size(-1) == 5:
                    bboxes = RotatedBoxes(bboxes)
                elif bboxes.size(-1) == 8:
                    bboxes = QuadriBoxes(bboxes)
                else:
                    raise TypeError(
                        'Require the shape of `bboxes` to be (n, 5) '
                        'or (n, 8), but get `bboxes` with shape being '
                        f'{bboxes.shape}.')

            bboxes = bboxes.cpu()
            polygons = bboxes.convert_to('qbox').tensor
            polygons = polygons.reshape(-1, 4, 2)
            polygons = [p for p in polygons]
            self.draw_polygons(
                polygons,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width)

            positions = bboxes.centers + self.line_width
            scales = _get_adaptive_scales(bboxes.areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = classes[
                    label] if classes is not None else f'class {label}'
                if 'scores' in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    label_text += f': {score}'

                self.draw_texts(
                    label_text,
                    pos,
                    colors=text_colors[i],
                    font_sizes=int(13 * scales[i]),
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        if 'masks' in instances:
            labels = instances.labels
            masks = instances.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            masks = masks.astype(bool)

            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None \
                else self.mask_color
            mask_palette = get_palette(mask_color, max_label + 1)
            colors = [jitter_color(mask_palette[label]) for label in labels]
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)
        return self.get_image()

@METRICS.register_module()
class DOTAMetric(BaseMetric):
    """DOTA evaluation metric.

    Note:  In addition to format the output results to JSON like CocoMetric,
    it can also generate the full image's results by merging patches' results.
    The premise is that you must use the tool provided by us to crop the DOTA
    large images, which can be found at: ``tools/data/dota/split``.

    Args:
        iou_thrs (float or List[float]): IoU threshold. Defaults to 0.5.
        scale_ranges (List[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None.
        metric (str | list[str]): Metrics to be evaluated. Only support
            'mAP' now. If is list, the first setting in the list will
             be used to evaluate metric.
        predict_box_type (str): Box type of model results. If the QuadriBoxes
            is used, you need to specify 'qbox'. Defaults to 'rbox'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format. Defaults to False.
        outfile_prefix (str, optional): The prefix of json/zip files. It
            includes the file path and the prefix of filename, e.g.,
            "a/b/prefix". If not specified, a temp file will be created.
            Defaults to None.
        merge_patches (bool): Generate the full image's results by merging
            patches' results.
        iou_thr (float): IoU threshold of ``nms_rotated`` used in merge
            patches. Defaults to 0.1.
        eval_mode (str): 'area' or '11points', 'area' means calculating the
            area under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1].
            The PASCAL VOC2007 defaults to use '11points', while PASCAL
            VOC2012 defaults to use 'area'. Defaults to '11points'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix: Optional[str] = 'dota'

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[tuple]] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 predict_box_type: str = 'rbox',
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 merge_patches: bool = False,
                 iou_thr: float = 0.1,
                 eval_mode: str = '11points',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
            else iou_thrs
        assert isinstance(self.iou_thrs, list)
        self.scale_ranges = scale_ranges
        # voc evaluation metrics
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f"metric should be one of 'mAP', but got {metric}.")
        self.metric = metric
        self.predict_box_type = predict_box_type

        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix
        self.merge_patches = merge_patches
        self.iou_thr = iou_thr

        self.use_07_metric = True if eval_mode == '11points' else False

    def merge_results(self, results: Sequence[dict],
                      outfile_prefix: str) -> str:
        """Merge patches' predictions into full image's results and generate a
        zip file for DOTA online evaluation.

        You can submit it at:
        https://captain-whu.github.io/DOTA/evaluation.html

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the zip files. If the
                prefix is "somepath/xxx", the zip files will be named
                "somepath/xxx/xxx.zip".
        """
        collector = defaultdict(list)

        for idx, result in enumerate(results):
            img_id = result.get('img_id', idx)
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            ori_bboxes = bboxes.copy()
            if self.predict_box_type == 'rbox':
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
            elif self.predict_box_type == 'qbox':
                ori_bboxes[..., :] = ori_bboxes[..., :] + np.array(
                    [x, y, x, y, x, y, x, y], dtype=np.float32)
            else:
                raise NotImplementedError
            label_dets = np.concatenate(
                [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
                axis=1)
            collector[oriname].append(label_dets)

        id_list, dets_list = [], []
        for oriname, label_dets_list in collector.items():
            big_img_results = []
            label_dets = np.concatenate(label_dets_list, axis=0)
            labels, dets = label_dets[:, 0], label_dets[:, 1:]
            for i in range(len(self.dataset_meta['classes'])):
                if len(dets[labels == i]) == 0:
                    big_img_results.append(dets[labels == i])
                else:
                    try:
                        cls_dets = torch.from_numpy(dets[labels == i]).cuda()
                    except:  # noqa: E722
                        cls_dets = torch.from_numpy(dets[labels == i])
                    if self.predict_box_type == 'rbox':
                        nms_dets, _ = nms_rotated(cls_dets[:, :5],
                                                  cls_dets[:,
                                                           -1], self.iou_thr)
                    elif self.predict_box_type == 'qbox':
                        nms_dets, _ = nms_quadri(cls_dets[:, :8],
                                                 cls_dets[:, -1], self.iou_thr)
                    else:
                        raise NotImplementedError
                    big_img_results.append(nms_dets.cpu().numpy())
            id_list.append(oriname)
            dets_list.append(big_img_results)

        if osp.exists(outfile_prefix):
            raise ValueError(f'The outfile_prefix should be a non-exist path, '
                             f'but {outfile_prefix} is existing. '
                             f'Please delete it firstly.')
        os.makedirs(outfile_prefix)

        files = [
            osp.join(outfile_prefix, 'Task1_' + cls + '.txt')
            for cls in self.dataset_meta['classes']
        ]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0:
                    continue
                th_dets = torch.from_numpy(dets)
                if self.predict_box_type == 'rbox':
                    rboxes, scores = torch.split(th_dets, (5, 1), dim=-1)
                    qboxes = rbox2qbox(rboxes)
                elif self.predict_box_type == 'qbox':
                    qboxes, scores = torch.split(th_dets, (8, 1), dim=-1)
                else:
                    raise NotImplementedError
                for qbox, score in zip(qboxes, scores):
                    txt_element = [img_id, str(round(float(score), 2))
                                   ] + [f'{p:.2f}' for p in qbox]
                    f.writelines(' '.join(txt_element) + '\n')

        for f in file_objs:
            f.close()

        target_name = osp.split(outfile_prefix)[-1]
        zip_path = osp.join(outfile_prefix, target_name + '.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])

        return zip_path

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = bboxes[i].tolist()
                data['score'] = float(scores[i])
                data['category_id'] = int(label)
                bbox_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        return result_files

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            if gt_instances == {}:
                ann = dict()
            else:
                ann = dict(
                    labels=gt_instances['labels'].cpu().numpy(),
                    bboxes=gt_instances['bboxes'].cpu().numpy(),
                    bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                    labels_ignore=gt_ignore_instances['labels'].cpu().numpy())
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()

            result['pred_bbox_scores'] = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(result['labels'] == label)[0]
                pred_bbox_scores = np.hstack([
                    result['bboxes'][index], result['scores'][index].reshape(
                        (-1, 1))
                ])
                result['pred_bbox_scores'].append(pred_bbox_scores)

            self.results.append((ann, result))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()
        if self.merge_patches:
            # convert predictions to txt format and dump to zip file
            zip_path = self.merge_results(preds, outfile_prefix)
            logger.info(f'The submission file save at {zip_path}')
            return eval_results
        else:
            # convert predictions to coco format and dump to json file
            _ = self.results2json(preds, outfile_prefix)
            if self.format_only:
                logger.info('results are saved in '
                            f'{osp.dirname(outfile_prefix)}')
                return eval_results

        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            dataset_name = self.dataset_meta['classes']
            dets = [pred['pred_bbox_scores'] for pred in preds]

            mean_aps = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_rbbox_map(
                    dets,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=self.use_07_metric,
                    box_type=self.predict_box_type,
                    dataset=dataset_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        else:
            raise NotImplementedError
        return eval_results

@TRANSFORMS.register_module()
class RandomRotate(BaseTransform):
    """Random rotate image & bbox & masks. The rotation angle will choice in.

    [-angle_range, angle_range). Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)
    Modified Keys:
    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map
    Added Keys:
    - homography_matrix
    Args:
        prob (float): The probability of whether to rotate or not. Defaults
            to 0.5.
        angle_range (int): The maximum range of rotation angle. The rotation
            angle will lie in [-angle_range, angle_range). Defaults to 180.
        rect_obj_labels (List[int], Optional): A list of labels whose
            corresponding objects are alwags horizontal. If
            results['gt_bboxes_labels'] has any label in ``rect_obj_labels``,
            the rotation angle will only be choiced from [90, 180, -90, -180].
            Defaults to None.
        rotate_type (str): The type of rotate class to use. Defaults to
            "Rotate".
        **rotate_kwargs: Other keyword arguments for the ``rotate_type``.
    """

    def __init__(self,
                 prob: float = 0.5,
                 angle_range: int = 180,
                 rect_obj_labels: Optional[List[int]] = None,
                 rotate_type: str = 'Rotate',
                 **rotate_kwargs) -> None:
        assert 0 < angle_range <= 180
        self.prob = prob
        self.angle_range = angle_range
        self.rect_obj_labels = rect_obj_labels
        self.rotate_cfg = dict(type=rotate_type, **rotate_kwargs)
        self.rotate = TRANSFORMS.build({'rotate_angle': 0, **self.rotate_cfg})
        self.horizontal_angles = [90, 180, -90, -180]

    @cache_randomness
    def _random_angle(self) -> int:
        """Random angle."""
        return self.angle_range * (2 * np.random.rand() - 1)

    @cache_randomness
    def _random_horizontal_angle(self) -> int:
        """Random horizontal angle."""
        return np.random.choice(self.horizontal_angles)

    @cache_randomness
    def _is_rotate(self) -> bool:
        """Randomly decide whether to rotate."""
        return np.random.rand() < self.prob

    def transform(self, results: dict) -> dict:
        """The transform function."""
        if not self._is_rotate():
            return results

        rotate_angle = self._random_angle()
        if self.rect_obj_labels is not None and 'gt_bboxes_labels' in results:
            for label in self.rect_obj_labels:
                if (results['gt_bboxes_labels'] == label).any():
                    rotate_angle = self._random_horizontal_angle()
                    break

        self.rotate.rotate_angle = rotate_angle
        return self.rotate(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'rotate_angle={self.angle_range}, '
        repr_str += f'rect_obj_labels={self.rect_obj_labels}, '
        repr_str += f'rotate_cfg={self.rotate_cfg})'
        return repr_str

@TRANSFORMS.register_module()
class ConvertBoxType(BaseTransform):
    """Convert boxes in results to a certain box type.

    Args:
        box_type_mapping (dict): A dictionary whose key will be used to search
            the item in `results`, the value is the destination box type.
    """

    def __init__(self, box_type_mapping: dict) -> None:
        self.box_type_mapping = box_type_mapping

    def transform(self, results: dict) -> dict:
        """The transform function."""
        for key, dst_box_type in self.box_type_mapping.items():
            if key not in results:
                continue
            assert isinstance(results[key], BaseBoxes), \
                f"results['{key}'] not a instance of BaseBoxes."
            results[key] = results[key].convert_to(dst_box_type)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(box_type_mapping={self.box_type_mapping})'
        return repr_str

@TASK_UTILS.register_module()
class PseudoAngleCoder(BaseBBoxCoder):
    """Pseudo Angle Coder."""

    encode_size = 1

    def encode(self, angle_targets: Tensor) -> Tensor:
        return angle_targets

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        if keepdim:
            return angle_preds
        else:
            return angle_preds.squeeze(-1)
def norm_angle(angle, angle_range):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.

    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    elif angle_range == 'r360':
        return (angle + np.pi) % (2 * np.pi) - np.pi
    else:
        print('Not yet implemented.')


def gaussian2bbox(gmm):
    """Convert Gaussian distribution to polygons by SVD.

    Args:
        gmm (dict[str, torch.Tensor]): Dict of Gaussian distribution.

    Returns:
        torch.Tensor: Polygons.
    """
    try:
        from torch_batch_svd import svd
    except ImportError:
        svd = None
    L = 3
    var = gmm.var
    mu = gmm.mu
    assert mu.size()[1:] == (1, 2)
    assert var.size()[1:] == (1, 2, 2)
    T = mu.size()[0]
    var = var.squeeze(1)
    if svd is None:
        raise ImportError('Please install torch_batch_svd first.')
    U, s, Vt = svd(var)
    size_half = L * s.sqrt().unsqueeze(1).repeat(1, 4, 1)
    mu = mu.repeat(1, 4, 1)
    dx_dy = size_half * torch.tensor([[-1, 1], [1, 1], [1, -1], [-1, -1]],
                                     dtype=torch.float32,
                                     device=size_half.device)
    bboxes = (mu + dx_dy.matmul(Vt.transpose(1, 2))).reshape(T, 8)

    return bboxes


def gt2gaussian(target):
    """Convert polygons to Gaussian distributions.

    Args:
        target (torch.Tensor): Polygons with shape (N, 8).

    Returns:
        dict[str, torch.Tensor]: Gaussian distributions.
    """
    L = 3
    center = torch.mean(target, dim=1)
    edge_1 = target[:, 1, :] - target[:, 0, :]
    edge_2 = target[:, 2, :] - target[:, 1, :]
    w = (edge_1 * edge_1).sum(dim=-1, keepdim=True)
    w_ = w.sqrt()
    h = (edge_2 * edge_2).sum(dim=-1, keepdim=True)
    diag = torch.cat([w, h], dim=-1).diag_embed() / (4 * L * L)
    cos_sin = edge_1 / w_
    neg = torch.tensor([[1, -1]], dtype=torch.float32).to(cos_sin.device)
    R = torch.stack([cos_sin * neg, cos_sin[..., [1, 0]]], dim=-2)

    return (center, R.matmul(diag).matmul(R.transpose(-1, -2)))


def distance2obb(points: torch.Tensor,
                 distance: torch.Tensor,
                 angle_version: str = 'oc'):
    """Convert distance angle to rotated boxes.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries and angle (left, top, right, bottom, angle).
            Shape (B, N, 5) or (N, 5)
        angle_version: angle representations.
    Returns:
        dict[str, torch.Tensor]: Gaussian distributions.
    """
    distance, angle = distance.split([4, 1], dim=-1)

    cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)

    rot_matrix = torch.cat([cos_angle, -sin_angle, sin_angle, cos_angle],
                           dim=-1)
    rot_matrix = rot_matrix.reshape(*rot_matrix.shape[:-1], 2, 2)

    wh = distance[..., :2] + distance[..., 2:]
    offset_t = (distance[..., 2:] - distance[..., :2]) / 2
    offset_t = offset_t.unsqueeze(-1)
    offset = torch.matmul(rot_matrix, offset_t).squeeze(-1)
    ctr = points[..., :2] + offset

    angle_regular = norm_angle(angle, angle_version)
    return torch.cat([ctr, wh, angle_regular], dim=-1)

@weighted_loss
def rotated_iou_loss(pred, target, linear=False, mode='log', eps=1e-6):
    """Rotated IoU loss.

    Computing the IoU loss between a set of predicted rbboxes and target
     rbboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn(
            'DeprecationWarning: Setting "linear=True" in '
            'poly_iou_loss is deprecated, please use "mode=`linear`" '
            'instead.')

    if diff_iou_rotated_2d is None:
        raise ImportError('Please install mmcv-full >= 1.5.0.')

    ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    ious = ious.squeeze(0).clamp(min=eps)

    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss
@MODELS.register_module()
class RotatedIoULoss(nn.Module):
    """RotatedIoULoss.

    Computing the IoU loss between a set of predicted rbboxes and
    target rbboxes.
    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='log'):
        super(RotatedIoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 5) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.loss_weight * rotated_iou_loss(
                pred,
                target,
                weight,
                mode=self.mode,
                eps=self.eps,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
        return loss
@MODELS.register_module()
class RotatedRTMDetHead(RTMDetHead):
    """Detection Head of Rotated RTMDet.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        angle_version (str): Angle representations. Defaults to 'le90'.
        use_hbbox_loss (bool): If true, use horizontal bbox loss and
            loss_angle should not be None. Default to False.
        scale_angle (bool): If true, add scale to angle pred branch.
            Default to True.
        angle_coder (:obj:`ConfigDict` or dict): Config of angle coder.
        loss_angle (:obj:`ConfigDict` or dict, Optional): Config of angle loss.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 angle_version: str = 'le90',
                 use_hbbox_loss: bool = False,
                 scale_angle: bool = True,
                 angle_coder: ConfigType = dict(type='PseudoAngleCoder'),
                 loss_angle: OptConfigType = None,
                 **kwargs) -> None:
        self.angle_version = angle_version
        self.use_hbbox_loss = use_hbbox_loss
        self.is_scale_angle = scale_angle
        self.angle_coder = TASK_UTILS.build(angle_coder)
        super().__init__(
            num_classes,
            in_channels,
            # useless, but error
            loss_centerness=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0),
            **kwargs)
        if loss_angle is not None:
            self.loss_angle = MODELS.build(loss_angle)
        else:
            self.loss_angle = None

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        pred_pad_size = self.pred_kernel_size // 2
        self.rtm_ang = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.angle_coder.encode_size,
            self.pred_kernel_size,
            padding=pred_pad_size)
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super().init_weights()
        normal_init(self.rtm_ang, std=0.01)

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - angle_preds (list[Tensor]): Angle prediction for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * angle_dim.
        """

        cls_scores = []
        bbox_preds = []
        angle_preds = []
        for idx, (x, scale, stride) in enumerate(
                zip(feats, self.scales, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls(cls_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj(reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))

            reg_dist = scale(self.rtm_reg(reg_feat).exp()).float() * stride[0]
            if self.is_scale_angle:
                angle_pred = self.scale_angle(self.rtm_ang(reg_feat)).float()
            else:
                angle_pred = self.rtm_ang(reg_feat).float()

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(angle_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds)

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            angle_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            assign_metrics: Tensor, stride: List[int]):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 5, H, W) for rbox loss
                or (N, num_anchors * 4, H, W) for hbox loss.
            angle_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * angle_dim, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            assign_metrics (Tensor): Assign metrics with shape
                (N, num_total_anchors).
            stride (List[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()

        if self.use_hbbox_loss:
            bbox_pred = bbox_pred.reshape(-1, 4)
        else:
            bbox_pred = bbox_pred.reshape(-1, 5)
        bbox_targets = bbox_targets.reshape(-1, 5)

        labels = labels.reshape(-1)
        assign_metrics = assign_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, assign_metrics)

        loss_cls = self.loss_cls(
            cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets
            if self.use_hbbox_loss:
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(
                    pos_bbox_targets[:, :4])

            # regression loss
            pos_bbox_weight = assign_metrics[pos_inds]

            loss_angle = angle_pred.sum() * 0
            if self.loss_angle is not None:
                angle_pred = angle_pred.reshape(-1,
                                                self.angle_coder.encode_size)
                pos_angle_pred = angle_pred[pos_inds]
                pos_angle_target = pos_bbox_targets[:, 4:5]
                pos_angle_target = self.angle_coder.encode(pos_angle_target)
                if pos_angle_target.dim() == 2:
                    pos_angle_weight = pos_bbox_weight.unsqueeze(-1)
                else:
                    pos_angle_weight = pos_bbox_weight
                loss_angle = self.loss_angle(
                    pos_angle_pred,
                    pos_angle_target,
                    weight=pos_angle_weight,
                    avg_factor=1.0)

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)

        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)
            loss_angle = angle_pred.sum() * 0

        return (loss_cls, loss_bbox, loss_angle, assign_metrics.sum(),
                pos_bbox_weight.sum(), pos_bbox_weight.sum())

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     angle_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box predict for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [t, b, l, r] format.
            bbox_preds (list[Tensor]): Angle pred for each scale
                level with shape (N, num_anchors * angle_dim, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)

        decoded_bboxes = []
        decoded_hbboxes = []
        angle_preds_list = []
        for anchor, bbox_pred, angle_pred in zip(anchor_list[0], bbox_preds,
                                                 angle_preds):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.angle_coder.encode_size)

            if self.use_hbbox_loss:
                hbbox_pred = distance2bbox(anchor, bbox_pred)
                decoded_hbboxes.append(hbbox_pred)

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            bbox_pred = distance2obb(
                anchor, bbox_pred, angle_version=self.angle_version)
            decoded_bboxes.append(bbox_pred)
            angle_preds_list.append(angle_pred)

        # flatten_bboxes is rbox, for target assign
        flatten_bboxes = torch.cat(decoded_bboxes, 1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         assign_metrics_list, sampling_results_list) = cls_reg_targets

        if self.use_hbbox_loss:
            decoded_bboxes = decoded_hbboxes

        (losses_cls, losses_bbox, losses_angle, cls_avg_factors,
         bbox_avg_factors, angle_avg_factors) = multi_apply(
             self.loss_by_feat_single, cls_scores, decoded_bboxes,
             angle_preds_list, labels_list, label_weights_list,
             bbox_targets_list, assign_metrics_list,
             self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        if self.loss_angle is not None:
            angle_avg_factors = reduce_mean(
                sum(angle_avg_factors)).clamp_(min=1).item()
            losses_angle = list(
                map(lambda x: x / angle_avg_factors, losses_angle))
            return dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox,
                loss_angle=losses_angle)
        else:
            return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def _get_targets_single(self,
                            cls_scores: Tensor,
                            bbox_preds: Tensor,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            cls_scores (list(Tensor)): Box scores for each image.
            bbox_preds (list(Tensor)): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

        Returns:
            tuple: N is the number of total anchors in the image.

            - anchors (Tensor): All anchors in the image with shape (N, 4).
            - labels (Tensor): Labels of all anchors in the image with shape
              (N,).
            - label_weights (Tensor): Label weights of all anchor in the
              image with shape (N,).
            - bbox_targets (Tensor): BBox targets of all anchors in the
              image with shape (N, 5).
            - norm_alignment_metrics (Tensor): Normalized alignment metrics
              of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        pred_instances = InstanceData(
            scores=cls_scores[inside_flags, :],
            bboxes=bbox_preds[inside_flags, :],
            priors=anchors)

        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = anchors.new_zeros((*anchors.size()[:-1], 5))
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        assign_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            pos_bbox_targets = pos_bbox_targets.regularize_boxes(
                self.angle_version)
            bbox_targets[pos_inds, :] = pos_bbox_targets

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            assign_metrics[gt_class_inds] = assign_result.max_overlaps[
                gt_class_inds]

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            assign_metrics = unmap(assign_metrics, num_total_anchors,
                                   inside_flags)
        return (anchors, labels, label_weights, bbox_targets, assign_metrics,
                sampling_result)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.
        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            angle_preds (list[Tensor]): Box angle for each scale level
                with shape (N, num_points * angle_dim, H, W)
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.
        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 5 arrange as (x, y, w, h, t).
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            angle_pred_list = select_single_mlvl(
                angle_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                angle_pred_list=angle_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.
        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            angle_pred_list (list[Tensor]): Box angle for a single scale
                level with shape (N, num_points * angle_dim, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.
        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 5 arrange as (x, y, w, h, t).
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (
                cls_score, bbox_pred, angle_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred, angle_pred=angle_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            angle_pred = filtered_results['angle_pred']
            priors = filtered_results['priors']

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)



@TASK_UTILS.register_module()
class DistanceAnglePointCoder(BaseBBoxCoder):
    """Distance Angle Point BBox coder.

    This coder encodes gt bboxes (x, y, w, h, theta) into (top, bottom, left,
    right, theta) and decode it back to the original.

    Args:
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self, clip_border=True, angle_version='oc'):
        super(BaseBBoxCoder, self).__init__()
        self.clip_border = clip_border
        self.angle_version = angle_version

    def encode(self, points, gt_bboxes, max_dis=None, eps=0.1):
        """Encode bounding box to distances.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            gt_bboxes (Tensor): Shape (N, 5), The format is "xywha"
            max_dis (float): Upper bound of the distance. Default None.
            eps (float): a small value to ensure target < max_dis, instead <=.
                Default 0.1.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 5).
        """
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 5
        return self.obb2distance(points, gt_bboxes, max_dis, eps)

    def decode(self, points, pred_bboxes, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            pred_bboxes (Tensor): Distance from the given point to 4
                boundaries and angle (left, top, right, bottom, angle).
                Shape (B, N, 5) or (N, 5)
            max_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If priors shape is (B, N, 4), then
                the max_shape should be a Sequence[Sequence[int]],
                and the length of max_shape should also be B.
                Default None.
        Returns:
            Tensor: Boxes with shape (N, 5) or (B, N, 5)
        """
        assert points.size(0) == pred_bboxes.size(0)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 5
        if self.clip_border is False:
            max_shape = None
        return self.distance2obb(points, pred_bboxes, max_shape,
                                 self.angle_version)

    def obb2distance(self, points, distance, max_dis=None, eps=None):
        ctr, wh, angle = torch.split(distance, [2, 2, 1], dim=1)

        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=1).reshape(-1, 2, 2)

        offset = points - ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = wh[..., 0], wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        if max_dis is not None:
            left = left.clamp(min=0, max=max_dis - eps)
            top = top.clamp(min=0, max=max_dis - eps)
            right = right.clamp(min=0, max=max_dis - eps)
            bottom = bottom.clamp(min=0, max=max_dis - eps)
        return torch.stack((left, top, right, bottom, angle.squeeze(-1)), -1)

    def distance2obb(self,
                     points,
                     distance,
                     max_shape=None,
                     angle_version='oc'):
        distance, angle = distance.split([4, 1], dim=-1)

        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)

        rot_matrix = torch.cat([cos_angle, -sin_angle, sin_angle, cos_angle],
                               dim=-1)
        rot_matrix = rot_matrix.reshape(*rot_matrix.shape[:-1], 2, 2)

        wh = distance[..., :2] + distance[..., 2:]
        offset_t = (distance[..., 2:] - distance[..., :2]) / 2
        offset = torch.matmul(rot_matrix, offset_t[..., None]).squeeze(-1)
        ctr = points[..., :2] + offset

        angle_regular = norm_angle(angle, angle_version)
        return torch.cat([ctr, wh, angle_regular], dim=-1)


@MODELS.register_module()
class RotatedRTMDetSepBNHead(RotatedRTMDetHead):
    """Rotated RTMDetHead with separated BN layers and shared conv layers.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        share_conv (bool): Whether to share conv layers between stages.
            Defaults to True.
        scale_angle (bool): Does not support in RotatedRTMDetSepBNHead,
            Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict)): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict)): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        pred_kernel_size (int): Kernel size of prediction layer. Defaults to 1.
        exp_on_reg (bool): Whether to apply exponential on bbox_pred.
            Defaults to False.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 share_conv: bool = True,
                 scale_angle: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 pred_kernel_size: int = 1,
                 exp_on_reg: bool = False,
                 **kwargs) -> None:
        self.share_conv = share_conv
        self.exp_on_reg = exp_on_reg
        assert scale_angle is False, \
            'scale_angle does not support in RotatedRTMDetSepBNHead'
        super().__init__(
            num_classes,
            in_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            pred_kernel_size=pred_kernel_size,
            scale_angle=False,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_ang = nn.ModuleList()
        if self.with_objectness:
            self.rtm_obj = nn.ModuleList()
        for n in range(len(self.prior_generator.strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.cls_out_channels,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_ang.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.angle_coder.encode_size,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            if self.with_objectness:
                self.rtm_obj.append(
                    nn.Conv2d(
                        self.feat_channels,
                        1,
                        self.pred_kernel_size,
                        padding=self.pred_kernel_size // 2))

        if self.share_conv:
            for n in range(len(self.prior_generator.strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        for rtm_cls, rtm_reg, rtm_ang in zip(self.rtm_cls, self.rtm_reg,
                                             self.rtm_ang):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01)
            normal_init(rtm_ang, std=0.01)
        if self.with_objectness:
            for rtm_obj in self.rtm_obj:
                normal_init(rtm_obj, std=0.01, bias=bias_cls)

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - angle_preds (list[Tensor]): Angle prediction for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * angle_dim.
        """
        cls_scores = []
        bbox_preds = []
        angle_preds = []
        for idx, (x, stride) in enumerate(
                zip(feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))
            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
            else:
                reg_dist = self.rtm_reg[idx](reg_feat) * stride[0]

            angle_pred = self.rtm_ang[idx](reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(angle_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds)



@TASK_UTILS.register_module()
class RBboxOverlaps2D(object):
    """2D Overlaps Calculator for Rotated Bboxes."""

    def __call__(self,
                 bboxes1: RotatedBoxes,
                 bboxes2: RotatedBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False) -> Tensor:
        """Calculate IoU between 2D rotated bboxes.

        Args:
            bboxes1 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (m, 5)
                in <cx, cy, w, h, t> format, shape (m, 6) in
                <cx, cy, w, h, t, score> format.
            bboxes2 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (n, 5)
                in <cx, cy, w, h, t> format, shape (n, 6) in
                <cx, cy, w, h, t, score> format, or be empty.
            mode (str): 'iou' (intersection over union), 'iof' (intersection
                over foreground). Defaults to 'iou'.
            is_aligned (bool): If True, then m and n must be equal.
                Defaults to False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]

        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]

        bboxes1 = get_box_tensor(bboxes1)
        bboxes2 = get_box_tensor(bboxes2)

        return rbbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self) -> str:
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str






@DATASETS.register_module()
class DOTADataset(BaseDataset):
    """DOTA-v1.0 dataset for detection.

    Note: ``ann_file`` in DOTADataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTADataset,
    it is the path of a folder containing XML files.

    Args:
        diff_thr (int): The difficulty threshold of ground truth. Bboxes
            with difficulty higher than it will be ignored. The range of this
            value should be non-negative integer. Defaults to 100.
        img_suffix (str): The suffix of images. Defaults to 'png'.
    """

    METAINFO = {
        'classes':
        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255)]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'png',
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in '
                                 f'{self.ann_file}')
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)

                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        instance = {}
                        bbox_info = si.split()
                        instance['bbox'] = [float(i) for i in bbox_info[:8]]
                        cls_name = bbox_info[8]
                        instance['bbox_label'] = cls_map[cls_name]
                        difficulty = int(bbox_info[9])
                        if difficulty > self.diff_thr:
                            instance['ignore_flag'] = 1
                        else:
                            instance['ignore_flag'] = 0
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DOTA category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]


@DATASETS.register_module()
class DOTAv15Dataset(DOTADataset):
    """DOTA-v1.5 dataset for detection.

    Note: ``ann_file`` in DOTAv15Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv15Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter', 'container-crane'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255), (220, 20, 60)]
    }


@DATASETS.register_module()
class DOTAv2Dataset(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport',
         'helipad'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255), (220, 20, 60), (119, 11, 32),
                    (0, 0, 142)]
    }

def rbbox_overlaps(bboxes1: Tensor,
                   bboxes2: Tensor,
                   mode: str = 'iou',
                   is_aligned: bool = False) -> Tensor:
    """Calculate overlap between two set of rotated bboxes.

    Args:
        bboxes1 (Tensor): shape (B, m, 5) in <cx, cy, w, h, t> format
            or empty.
        bboxes2 (Tensor): shape (B, n, 5) in <cx, cy, w, h, t> format
            or empty.
        mode (str): 'iou' (intersection over union), 'iof' (intersection over
            foreground). Defaults to 'iou'.
        is_aligned (bool): If True, then m and n must be equal.
            Defaults to False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    # resolve `rbbox_overlaps` abnormal when input rbbox is too small.
    clamped_bboxes1 = bboxes1.detach().clone()
    clamped_bboxes2 = bboxes2.detach().clone()
    clamped_bboxes1[:, 2:4].clamp_(min=1e-3)
    clamped_bboxes2[:, 2:4].clamp_(min=1e-3)

    # resolve `rbbox_overlaps` abnormal when coordinate value is too large.
    # TODO: fix in mmcv
    clamped_bboxes1[:, :2].clamp_(min=-1e7, max=1e7)
    clamped_bboxes2[:, :2].clamp_(min=-1e7, max=1e7)

    return box_iou_rotated(clamped_bboxes1, clamped_bboxes2, mode, is_aligned)


def fake_rbbox_overlaps(bboxes1: RotatedBoxes,
                        bboxes2: RotatedBoxes,
                        mode: str = 'iou',
                        is_aligned: bool = False) -> Tensor:
    """Calculate overlap between two set of minimum circumscribed hbbs of rbbs.

    Args:
        bboxes1 (:obj:`RotatedBoxes`): shape (B, m, 5) in <cx, cy, w, h, t>
            format or empty.
        bboxes2 (:obj:`RotatedBoxes`): shape (B, n, 5) in <cx, cy, w, h, t>
            format or empty.
        mode (str): 'iou' (intersection over union), 'iof' (intersection over
            foreground).
            Defaults to 'iou'.
        is_aligned (bool): If True, then m and n must be equal.
            Defaults to False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.tensor.new(
            rows, 1) if is_aligned else bboxes1.tensor.new(rows, cols)

    # convert rbb to minimum circumscribed hbb in <cx, cy, w, h, t> format.
    fake_rbboxes1 = bboxes1.convert_to('hbox').convert_to('rbox')
    fake_rbboxes2 = bboxes2.convert_to('hbox').convert_to('rbox')

    # resolve `rbbox_overlaps` abnormal when input rbbox is too small.
    clamped_bboxes1 = fake_rbboxes1.detach().clone().tensor
    clamped_bboxes2 = fake_rbboxes2.detach().clone().tensor
    clamped_bboxes1[:, 2:4].clamp_(min=1e-3)
    clamped_bboxes2[:, 2:4].clamp_(min=1e-3)

    # resolve `rbbox_overlaps` abnormal when coordinate value is too large.
    # TODO: fix in mmcv
    clamped_bboxes1[:, :2].clamp_(min=-1e7, max=1e7)
    clamped_bboxes2[:, :2].clamp_(min=-1e7, max=1e7)

    return box_iou_rotated(clamped_bboxes1, clamped_bboxes2, mode, is_aligned)
