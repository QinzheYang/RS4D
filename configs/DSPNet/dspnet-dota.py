#_base_ = ['./_base_/default_runtime.py', './_base_/schedule_3x.py','./_base_/dota_rr.py']
default_scope = 'mmdet'
custom_imports = dict(imports=['mmdet.rsprompter'], allow_failed_imports=False)
#custom_imports = dict(imports=['mmdet.rotated_rtmdet'], allow_failed_imports=False)
#checkpoint = '/mnt/user/yangqinzhe/project/project_code/mm_panda_CoDETR/configs/rotated_rtmdet/rotated_rtmdet_l-100e-aug-dota-bc59fd88.pth'  # noqa
work_dir = '/mnt/dataset/yangqinzhe/mamba-pth/dspnet'


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    #visualization=dict(type='DetVisualizationHook', draw=True, interval=1, test_out_dir='vis_data')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
batch_size=1
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='RotLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]
num_classes=18
angle_version = 'le90'
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        boxtype2tensor=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P7',
        expand_ratio=1,
        deepen_factor=1.5,
        widen_factor=2,
        out_indices = (3, 4, 5),
        channel_attention=True,
        spp_kernel_sizes=(5, 9, 13, 17),
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[1024, 1536, 2048],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    bbox_head=dict(
        type='RotatedRTMDetSepBNHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        angle_version=angle_version,
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        use_hbbox_loss=False,
        scale_angle=False,
        loss_angle=None,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.DynamicSoftLabelAssigner',
            iou_calculator=dict(type='RBboxOverlaps2D'),
            topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000),
)



max_epochs = 400
base_lr = 0.004 / 16
interval = 1

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# dataset settings
dataset_type = 'DOTAv2Dataset'
data_root = '/mnt/dataset/yangqinzhe/Rsprompter-mamba/demo/split_ss_dota/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    #dict(type='RandomRotate',
    #    prob=0.5,
    #    angle_range=180,
    #    rect_obj_labels=[9, 11]),
    dict(
        type='Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
# batch_size = (2 GPUs) x (4 samples per GPU) = 8

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    pin_memory=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
# test_dataloader = dict(
#     batch_size=8,
#     num_workers=8,
#     persistent_workers=False,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path='test/images/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='DOTAMetric',
#     format_only=True,
#     merge_patches=True,
#     outfile_prefix='./work_dirs/rtmdet_r/Task1')
