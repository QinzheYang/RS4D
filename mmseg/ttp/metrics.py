from collections import OrderedDict
from typing import Optional, Sequence, Dict
import numpy as np
import torch
from mmengine import MMLogger, print_log
from mmengine.evaluator import BaseMetric
from prettytable import PrettyTable
from torchmetrics.functional.classification import multiclass_precision, multiclass_recall, multiclass_f1_score, \
	multiclass_jaccard_index, multiclass_accuracy, binary_accuracy
from opencd.registry import METRICS


from typing import Dict, Optional, Sequence
from collections import OrderedDict
import torch
import numpy as np
from prettytable import PrettyTable
from mmengine.logging import MMLogger, print_log
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from torchmetrics.functional.classification import (
    multiclass_precision,
    multiclass_recall,
    multiclass_f1_score,
    multiclass_jaccard_index,
    multiclass_accuracy,
)

@METRICS.register_module()
class CDMetric(BaseMetric):
    default_prefix: Optional[str] = 'cd_seg'

    def __init__(
        self,
        ignore_index: int = 255,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            # 获取模型预测的分割结果
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # 获取真实的分割标签
            gt_label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
            # 将结果保存到 self.results
            self.results.append((pred_label, gt_label))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        # 获取类别数量和类别名称
        num_classes = len(self.dataset_meta['classes'])
        class_names = self.dataset_meta['classes']

        # 确保类别数量为 4
        assert num_classes == 4, 'CDSegMetric supports 4 classes (0, 1, 2, 3).'

        logger: MMLogger = MMLogger.get_current_instance()
        pred_label, label = zip(*results)
        preds = torch.stack(pred_label, dim=0)
        target = torch.stack(label, dim=0)

        # 计算多分类指标
        multiclass_precision_ = multiclass_precision(
            preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index
        )
        multiclass_recall_ = multiclass_recall(
            preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index
        )
        multiclass_f1_score_ = multiclass_f1_score(
            preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index
        )
        multiclass_jaccard_index_ = multiclass_jaccard_index(
            preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index
        )
        accuracy_ = multiclass_accuracy(
            preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index
        )

        # 计算 mIoU
        mIoU = multiclass_jaccard_index_.mean().item()

        # 将指标结果保存到 OrderedDict
        ret_metrics = OrderedDict({
            'acc': accuracy_.cpu().numpy(),
            'p': multiclass_precision_.cpu().numpy(),
            'r': multiclass_recall_.cpu().numpy(),
            'f1': multiclass_f1_score_.cpu().numpy(),
            'iou': multiclass_jaccard_index_.cpu().numpy(),
        })

        # 将指标结果转换为字典格式
        metrics = dict()
        for k, v in ret_metrics.items():
            for i in range(num_classes):
                metrics[k + '_' + class_names[i]] = v[i].item()
        metrics['mIoU'] = mIoU  # 单独添加 mIoU

        # 打印每个类别的指标结果
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # 创建表格并添加列
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        # 单独打印 mIoU
        print_log(f'mIoU: {mIoU * 100:.2f}', logger=logger)

        return metrics
    
@METRICS.register_module()
class CDMetric2(BaseMetric):
    default_prefix: Optional[str] = 'cd_seg'

    def __init__(
        self,
        ignore_index: int = 255,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            # 获取模型预测的分割结果
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # 获取真实的分割标签
            gt_label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
            # 将结果保存到 self.results
            self.results.append((pred_label, gt_label))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        # 获取类别数量和类别名称
        num_classes = len(self.dataset_meta['classes'])
        class_names = self.dataset_meta['classes']

        # 确保类别数量为 4
        assert num_classes == 4, 'CDSegMetric supports 4 classes (0, 1, 2, 3).'

        logger: MMLogger = MMLogger.get_current_instance()
        pred_label, label = zip(*results)
        preds = torch.stack(pred_label, dim=0)
        target = torch.stack(label, dim=0)

        # 计算多分类指标
        multiclass_precision_ = multiclass_precision(
            preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index
        )
        multiclass_recall_ = multiclass_recall(
            preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index
        )
        multiclass_f1_score_ = multiclass_f1_score(
            preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index
        )
        multiclass_jaccard_index_ = multiclass_jaccard_index(
            preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index
        )
        accuracy_ = multiclass_accuracy(
            preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index
        )

        # 将指标结果保存到 OrderedDict
        ret_metrics = OrderedDict({
            'acc': accuracy_.cpu().numpy(),
            'p': multiclass_precision_.cpu().numpy(),
            'r': multiclass_recall_.cpu().numpy(),
            'f1': multiclass_f1_score_.cpu().numpy(),
            'iou': multiclass_jaccard_index_.cpu().numpy(),
        })

        # 将指标结果转换为字典格式
        metrics = dict()
        for k, v in ret_metrics.items():
            for i in range(num_classes):
                metrics[k + '_' + class_names[i]] = v[i].item()

        # 打印每个类别的指标结果
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        return metrics

@METRICS.register_module()
class CD1Metric(BaseMetric):

	default_prefix: Optional[str] = 'cd'

	def __init__(self,
				 ignore_index: int = 255,
				 collect_device: str = 'cpu',
				 prefix: Optional[str] = None,
				 **kwargs) -> None:
		super().__init__(collect_device=collect_device, prefix=prefix)
		self.ignore_index = ignore_index

	def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
		for data_sample in data_samples:
			pred_label = data_sample['pred_sem_seg']['data'].squeeze()
			# format_only always for test dataset without ground truth
			gt_label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
			self.results.append((pred_label, gt_label))

	def compute_metrics(self, results: list) -> Dict[str, float]:
		num_classes = len(self.dataset_meta['classes'])
		class_names = self.dataset_meta['classes']

		assert num_classes == 2, 'Only support binary classification in CDMetric.'

		logger: MMLogger = MMLogger.get_current_instance()
		pred_label, label = zip(*results)
		preds = torch.stack(pred_label, dim=0)
		target = torch.stack(label, dim=0)

		multiclass_precision_ = multiclass_precision(preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index)
		multiclass_recall_ = multiclass_recall(preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index)
		multiclass_f1_score_ = multiclass_f1_score(preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index)
		multiclass_jaccard_index_ = multiclass_jaccard_index(preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index)
		accuracy_ = multiclass_accuracy(preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index)
		binary_accuracy_ = binary_accuracy(preds, target, ignore_index=self.ignore_index)
		ret_metrics = OrderedDict({
			'acc': accuracy_.cpu().numpy(),
			'p': multiclass_precision_.cpu().numpy(),
			'r': multiclass_recall_.cpu().numpy(),
			'f1': multiclass_f1_score_.cpu().numpy(),
			'iou': multiclass_jaccard_index_.cpu().numpy(),
			'macc': binary_accuracy_.cpu().numpy(),
		})

		metrics = dict()
		for k, v in ret_metrics.items():
			if k == 'macc':
				metrics[k] = v.item()
			else:
				for i in range(num_classes):
					metrics[k + '_' + class_names[i]] = v[i].item()

		# each class table
		ret_metrics.pop('macc', None)
		ret_metrics_class = OrderedDict({
			ret_metric: np.round(ret_metric_value * 100, 2)
			for ret_metric, ret_metric_value in ret_metrics.items()
		})
		ret_metrics_class.update({'Class': class_names})
		ret_metrics_class.move_to_end('Class', last=False)
		class_table_data = PrettyTable()
		for key, val in ret_metrics_class.items():
			class_table_data.add_column(key, val)

		print_log('per class results:', logger)
		print_log('\n' + class_table_data.get_string(), logger=logger)
		return metrics
