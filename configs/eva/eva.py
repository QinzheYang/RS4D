
#from mmpretrain.models.selfsup import EVA
#from mmpretrain.models.selfsup import MAEViT
#from mmpretrain.models.necks import MAEPretrainDecoder
#from mmpretrain.models.heads import MILANPretrainHead
#from mmpretrain.models.selfsup import CLIPGenerator
custom_imports = dict(imports=['mmdet.EVA'], allow_failed_imports=False)
default_scope = 'mmdet'
crop_size = (224, 224)
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]
work_dir = '/mnt/dataset/yangqinzhe/panda-pth/eva'
pretrained = '/mnt/user/yangqinzhe/project/project_code/mm_panda_CoDETR/swin_tiny_patch4_window7_224.pth'  # noqa
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=4, save_best='coco/bbox_mAP', rule='greater', save_last=True),
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
resume = False#True
# model settings
num_dec_layer = 6
loss_lambda = 2.0
num_classes = 1
image_size = (224, 224)
#image_size = (512, 384)
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
    batch_augments=batch_augments
)
# batch_augments = [
#     dict(type='BatchFixedSizePad', size=image_size, pad_mask=False)
# ]

model = dict(
    type='EVA1',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MAEViT1',
        arch='b',
        patch_size=16,
        mask_ratio=0.75,
        init_cfg=[
            dict(type='Xavier', distribution='uniform', layer='Linear'),
            dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
        ]),
    neck=dict(
        type='MAEPretrainDecoder1',
        predict_feature_dim=512,
        init_cfg=[
            dict(type='Xavier', distribution='uniform', layer='Linear'),
            dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
        ]),
    head=dict(
        type='MILANPretrainHead1',
        loss=dict(
            type='CosineSimilarityLoss', shift_factor=2.0, scale_factor=2.0),
    ),
    target_generator=dict(
        type='CLIPGenerator1',
        tokenizer_path=  # noqa
        '/mnt/user/yangqinzhe/project/project_code/RSPrompter-mamba/clip_vit_base_16.pth.tar'  # noqa
    ),
    init_cfg=None)


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
# optimizer

