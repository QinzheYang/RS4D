auto_scale_lr = dict(base_batch_size=4, enable=False)
base_lr = 0.0001
batch_augments = [
    dict(
        img_pad_value=0,
        mask_pad_value=0,
        pad_mask=True,
        pad_seg=False,
        size=(
            1024,
            1024,
        ),
        type='BatchFixedSizePad'),
]
batch_size = 4
batch_size_per_gpu = 1
code_root = '/root/autodl-tmp/project/project_code/workspace_pytorch/RSPrompter-mamba'
crop_size = (
    1024,
    1024,
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmdet.rsprompter',
    ])
data_preprocessor = dict(
    batch_augments=[
        dict(
            img_pad_value=0,
            mask_pad_value=0,
            pad_mask=True,
            pad_seg=False,
            size=(
                1024,
                1024,
            ),
            type='BatchFixedSizePad'),
    ],
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_mask=True,
    pad_size_divisor=32,
    std=[
        58.395,
        57.120000000000005,
        57.375,
    ],
    type='DetDataPreprocessor')
data_root = '/mnt/dataset/yangqinzhe/Rsprompter-mamba/NWPU'
dataset_type = 'NWPUInsSegDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=4,
        rule='greater',
        save_best='coco/bbox_mAP',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
hf_sam_pretrain_ckpt_path = 'work_dirs/sam_cache/sam_vit_base/pytorch_model.bin'
hf_sam_pretrain_name = 'work_dirs/sam_cache/sam_vit_base'
indices = None
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 400
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(
            checkpoint='work_dirs/resnet/resnet50.pth',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='caffe',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=80,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=80,
            num_convs=4,
            type='FCNMaskHead'),
        mask_iou_head=dict(
            conv_out_channels=256,
            fc_out_channels=1024,
            in_channels=256,
            num_classes=80,
            num_convs=4,
            num_fcs=2,
            roi_feat_size=14,
            type='MaskIoUHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='MaskScoringRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=28,
            mask_thr_binary=0.5,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='MaskScoringRCNN')
num_classes = 10
num_workers = 8
optim_wrapper = dict(
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=50, start_factor=0.001, type='LinearLR'),
    dict(
        T_max=400,
        begin=1,
        by_epoch=True,
        end=400,
        eta_min=1.0000000000000001e-07,
        type='CosineAnnealingLR'),
]
persistent_workers = True
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/mnt/dataset/yangqinzhe/Rsprompter-mamba/NWPU/annotations/NWPU_instances_val.json',
        data_prefix=dict(img='imgs'),
        data_root='/mnt/dataset/yangqinzhe/Rsprompter-mamba/NWPU',
        indices=None,
        pipeline=[
            dict(to_float32=True, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    103.53,
                    116.28,
                    123.675,
                ), masks=0),
                size=(
                    1024,
                    1024,
                ),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        ttype='NWPUInsSegDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    format_only=False, metric=[
        'bbox',
        'segm',
    ], type='CocoMetric')
test_pipeline = [
    dict(to_float32=True, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(
        pad_val=dict(img=(
            103.53,
            116.28,
            123.675,
        ), masks=0),
        size=(
            1024,
            1024,
        ),
        type='Pad'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=400, type='EpochBasedTrainLoop', val_interval=2)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/mnt/dataset/yangqinzhe/Rsprompter-mamba/NWPU/annotations/NWPU_instances_train.json',
        data_prefix=dict(img='imgs'),
        data_root='/mnt/dataset/yangqinzhe/Rsprompter-mamba/NWPU',
        indices=None,
        pipeline=[
            dict(to_float32=True, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                resize_type='Resize',
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(
                allow_negative_crop=True,
                crop_size=(
                    1024,
                    1024,
                ),
                crop_type='absolute',
                recompute_bbox=True,
                type='RandomCrop'),
            dict(
                by_mask=True,
                min_gt_bbox_wh=(
                    1e-05,
                    1e-05,
                ),
                type='FilterAnnotations'),
            dict(type='PackDetInputs'),
        ],
        type='NWPUInsSegDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(to_float32=True, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        resize_type='Resize',
        scale=(
            1024,
            1024,
        ),
        type='RandomResize'),
    dict(
        allow_negative_crop=True,
        crop_size=(
            1024,
            1024,
        ),
        crop_type='absolute',
        recompute_bbox=True,
        type='RandomCrop'),
    dict(
        by_mask=True,
        min_gt_bbox_wh=(
            1e-05,
            1e-05,
        ),
        type='FilterAnnotations'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/mnt/dataset/yangqinzhe/Rsprompter-mamba/NWPU/annotations/NWPU_instances_val.json',
        data_prefix=dict(img='imgs'),
        data_root='/mnt/dataset/yangqinzhe/Rsprompter-mamba/NWPU',
        indices=None,
        pipeline=[
            dict(to_float32=True, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    103.53,
                    116.28,
                    123.675,
                ), masks=0),
                size=(
                    1024,
                    1024,
                ),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='NWPUInsSegDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    format_only=False, metric=[
        'bbox',
        'segm',
    ], type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/mnt/dataset/yangqinzhe/Rsprompter-mamba-pth/new/work_dirs/rsprompter-test/samseg-maskrcnn-mamba-nwpu-msrcnn2'
