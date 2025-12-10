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
batch_size_per_gpu = 4
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
data_root = '/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD'
dataset_type = 'SSDDInsSegDataset'
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
max_epochs = 800
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='work_dirs/resnet/resnet50.pth', type='Pretrained'),
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
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
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    mask_head=dict(
        cls_down_index=0,
        feat_channels=512,
        in_channels=256,
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_mask=dict(loss_weight=3.0, type='DiceLoss', use_sigmoid=True),
        mask_feature_head=dict(
            end_level=3,
            feat_channels=128,
            mask_stride=4,
            norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
            out_channels=256,
            start_level=0),
        num_classes=80,
        num_grids=[
            40,
            36,
            24,
            16,
            12,
        ],
        pos_scale=0.2,
        scale_ranges=(
            (
                1,
                96,
            ),
            (
                48,
                192,
            ),
            (
                96,
                384,
            ),
            (
                192,
                768,
            ),
            (
                384,
                2048,
            ),
        ),
        stacked_convs=4,
        strides=[
            8,
            8,
            16,
            32,
            32,
        ],
        type='SOLOV2Head'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=0,
        type='FPN'),
    test_cfg=dict(
        filter_thr=0.05,
        kernel='gaussian',
        mask_thr=0.5,
        max_per_img=100,
        nms_pre=500,
        score_thr=0.1,
        sigma=2.0),
    type='SOLOv2')
num_classes = 1
num_workers = 8
optim_wrapper = dict(
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=50, start_factor=0.001, type='LinearLR'),
    dict(
        T_max=800,
        begin=1,
        by_epoch=True,
        end=800,
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
        '/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD/annotations/SSDD_instances_val.json',
        data_prefix=dict(img='imgs'),
        data_root='/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD',
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
        type='SSDDInsSegDataset'),
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
train_cfg = dict(max_epochs=800, type='EpochBasedTrainLoop', val_interval=2)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD/annotations/SSDD_instances_train.json',
        data_prefix=dict(img='imgs'),
        data_root='/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD',
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
        type='SSDDInsSegDataset'),
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
        '/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD/annotations/SSDD_instances_val.json',
        data_prefix=dict(img='imgs'),
        data_root='/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD',
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
        type='SSDDInsSegDataset'),
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
work_dir = '/mnt/dataset/yangqinzhe/Rsprompter-mamba-pth/new/work_dirs/rsprompter-test/samseg-maskrcnn-mamba-ssdd-solov2'
