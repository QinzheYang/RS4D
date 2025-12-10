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
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5, save_best='coco/bbox_mAP', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    #visualization=dict(type='DetVisualizationHook', draw=True, interval=1, test_out_dir='vis_data')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
crop_size = (512, 512)
batch_size=2
vis_backends = [dict(type='LocalVisBackend'),]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
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
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
    batch_augments=batch_augments
)

num_classes=1
model = dict(
    type='DSPDet2D',
    data_preprocessor=data_preprocessor,
    backbone=dict(type='DSPBackbone2D', in_channels=3, max_channels=128, depth=34,  pool=False),
    head=dict(
        type='DSPHeadImage2',
        in_channels=(64, 128, 128, 128),#64 128 128
        out_channels=128,
        n_reg_outs=4,
        n_classes=1,
        assigner=dict(
            type='DSPAssignerImage',
            top_pts_threshold=6,
        ),
        prune_threshold=0.3,
        bbox_loss=dict(type='L1Loss', loss_weight=5.0)),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))

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
dataset_type = 'SSDDInsSegDataset'
data_root = '/mnt/dataset/yangqinzhe/Rsprompter-mamba/levir-ship/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]


test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0)),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]


val_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0)),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    #batch_sampler=None,
    #pin_memory=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        indices=None,
        ann_file='/mnt/dataset/yangqinzhe/Rsprompter-mamba/levir-ship/annotations/instances_train2017.json',
        data_prefix=dict(img='images/train2017/'),
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
        indices=None,
        data_root=data_root,
        ann_file='/mnt/dataset/yangqinzhe/Rsprompter-mamba/levir-ship/annotations/instances_val2017.json',
        data_prefix=dict(img='images/val2017/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        indices=None,
        data_root=data_root,
        ann_file='/mnt/dataset/yangqinzhe/Rsprompter-mamba/levir-ship/annotations/instances_test2017.json',
        data_prefix=dict(img='images/test2017/'),
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox'],
    format_only=False,
)
test_evaluator = val_evaluator

#val_evaluator = dict(type='DOTAMetric', metric='mAP')
#test_evaluator = val_evaluator

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
