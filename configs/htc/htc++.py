custom_imports = dict(imports=['mmdet.EVA'], allow_failed_imports=False)
default_scope = 'mmdet'
crop_size = (256, 256)
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]
work_dir = '/mnt/dataset/yangqinzhe/panda-pth/htc++_ssdd'
pretrained = '/mnt/user/yangqinzhe/project/project_code/mm_panda_CoDETR/swin_tiny_patch4_window7_224.pth'  # noqa
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2, save_best='coco/bbox_mAP', rule='greater', save_last=True),
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
pretrained = '/mnt/user/yangqinzhe/project/project_code/mm_panda_CoDETR/swin_tiny_patch4_window7_224.pth'  # noqa

log_level = 'INFO'
load_from = None
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa
resume = True
# model settings
num_dec_layer = 6
loss_lambda = 2.0
num_classes = 1
image_size = (256, 256)
model = dict(
    type='HybridTaskCascade',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='SwinTransformerV2',
        # embed_dims=96,
        arch='huge',
        #depths=[2, 2, 6, 2],
        #num_heads=[3, 6, 12, 24],
        window_size=7,
        #mlp_ratio=4,
        #qkv_bias=True,
        #qk_scale=None,
        drop_rate=0.,
        #attn_drop_rate=0.,
        drop_path_rate=0.2,
        #patch_norm=True,
        #out_indices=(0, 1, 2, 3),
        #with_cp=False,
        #convert_weights=True,
        #init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        type='FPN',
        in_channels=[352, 704, 1408, 2816],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='HybridTaskCascadeRoIHead',
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=num_classes,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=num_classes,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=num_classes,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

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
batch_size = 8
num_workers = 8
persistent_workers = True

data_root = '/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD'
dataset_type = 'SSDDInsSegDataset'

backend_args = None


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        indices=None,
        ann_file=
        '/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD/annotations/SSDD_instances_train.json',
        data_prefix=dict(img='imgs'),
        data_root='/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD',
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
        ann_file=
        '/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD/annotations/SSDD_instances_val.json',
        data_prefix=dict(img='imgs'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    format_only=False,
)
test_evaluator = val_evaluator

base_lr = 1e-4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

max_epochs = 400
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
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