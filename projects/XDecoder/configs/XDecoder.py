model = dict(
    type='XDecoder',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(type='FocalNet'),
    head=dict(
        type='XDecoderUnifiedhead',
        in_channels=(96, 192, 384, 768),
        pixel_decoder=dict(type='XTransformerEncoderPixelDecoder'),
        transformer_decoder=dict(type='XDecoderTransformerDecoder'),
        task='semseg',
    ),
    # use_thr_for_mc=True means use threshold for multi-class
    # This parameter is only used in semantic segmentation task and
    # referring semantic segmentation task.
    test_cfg=dict(mask_thr=0.5, use_thr_for_mc=True, ignore_index=255),
)
auto_scale_lr = dict(base_batch_size=1, enable=False)
base_lr = 0.0001
batch_augments = [
    dict(
        img_pad_value=0,
        mask_pad_value=0,
        pad_mask=True,
        pad_seg=False,
        size=(
            224,
            224,
        ),
        type='BatchFixedSizePad'),
]
batch_size = 1
batch_size_per_gpu = 1
code_root = '/root/autodl-tmp/project/project_code/workspace_pytorch/RSPrompter-mamba'
crop_size = (
    384,
    384,
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.XDecoder.xdecoder',
    ])
data_preprocessor = dict(
    batch_augments=[
        dict(
            img_pad_value=0,
            mask_pad_value=0,
            pad_mask=True,
            pad_seg=False,
            size=(
                384,
                384,
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
    batch_size=1,
    dataset=dict(
        ann_file=
        '/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD/annotations/SSDD_instances_val.json',
        data_prefix=dict(img='imgs'),
        data_root='/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD',
        indices=None,
        pipeline=[
            dict(to_float32=True, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                384,
                384,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    103.53,
                    116.28,
                    123.675,
                ), masks=0),
                size=(
                    384,
                    384,
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
        384,
        384,
    ), type='Resize'),
    dict(
        pad_val=dict(img=(
            103.53,
            116.28,
            123.675,
        ), masks=0),
        size=(
            384,
            384,
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
    batch_size=2,
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
                    384,
                    384,
                ),
                type='RandomResize'),
            dict(
                allow_negative_crop=True,
                crop_size=(
                    384,
                    384,
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
            384,
            384,
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
    batch_size=1,
    dataset=dict(
        ann_file=
        '/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD/annotations/SSDD_instances_val.json',
        data_prefix=dict(img='imgs'),
        data_root='/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD',
        indices=None,
        pipeline=[
            dict(to_float32=True, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                384,
                384,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    103.53,
                    116.28,
                    123.675,
                ), masks=0),
                size=(
                    384,
                    384,
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
work_dir = '/mnt/dataset/yangqinzhe/panda-pth/xdecoder'
