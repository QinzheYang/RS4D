custom_imports = dict(
    imports=['projects.SparseInst.sparseinst'], allow_failed_imports=False)
pretrained = '/mnt/user/yangqinzhe/project/project_code/RSPrompter-mamba/modified_model.pth'  # noqa
#pretrained = '/mnt/user/yangqinzhe/project/project_code/mm_panda_CoDETR/resnet50-0676ba61.pth'  # noqa
num_classes = 1
model = dict(
    type='SparseInst',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    encoder=dict(
        type='InstanceContextEncoder',
        in_channels=[512, 1024, 2048],
        out_channels=256),
    decoder=dict(
        type='BaseIAMDecoder',
        in_channels=256 + 2,
        num_classes=num_classes,
        ins_dim=256,
        ins_conv=4,
        mask_dim=256,
        mask_conv=4,
        kernel_dim=128,
        scale_factor=2.0,
        output_iam=False,
        num_masks=100),
    criterion=dict(
        type='SparseInstCriterion',
        num_classes=num_classes,
        assigner=dict(type='SparseInstMatcher', alpha=0.8, beta=0.2),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            alpha=0.25,
            gamma=2.0,
            reduction='sum',
            loss_weight=2.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            reduction='sum',
            eps=5e-5,
            loss_weight=2.0),
    ),
    test_cfg=dict(score_thr=0.005, mask_thr_binary=0.45))

default_scope = 'mmdet'
crop_size = (512, 512)

batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]
work_dir = '/mnt/dataset/yangqinzhe/panda-pth/sparseinst'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=4, save_best='coco/segm_mAP', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='DetVisualizationHook', draw=True, interval=1, test_out_dir='vis_data')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [
    dict(type='LocalVisBackend'),
    #dict(type='WandbVisBackend', init_kwargs=dict(project='panda', group='co_dino_swin_l', name='co_dino_swin_l_8G'))
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer', line_width=20)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa
resume = False
# model settings
num_dec_layer = 6
loss_lambda = 2.0

image_size = (512, 512)
#image_size = (512, 384)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True,with_mask=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2),by_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]


test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0)),
    dict(type='LoadAnnotations', with_bbox=True,with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
batch_size = 16
num_workers = 8
persistent_workers = True

dataset_type = 'SSDDInsSegDataset'

#### should be changed align with your code root and data root
code_root = '/root/autodl-tmp/project/project_code/workspace_pytorch/RSPrompter-mamba'
data_root = '/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD'
backend_args = None


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        indices=None,
        data_root=data_root,
        ann_file='/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD/annotations/SSDD_instances_train.json',
        data_prefix=dict(img='imgs'),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        indices=None,
        data_root=data_root,
        ann_file='/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD/annotations/SSDD_instances_val.json',
        data_prefix=dict(img='imgs'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    metric=['segm'],
    format_only=False,
)
test_evaluator = val_evaluator

base_lr = 1e-4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.05))

max_epochs = 8000
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=150),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.001,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True
    )
]
auto_scale_lr = dict(enable=False, base_batch_size=batch_size)
# optimizer

