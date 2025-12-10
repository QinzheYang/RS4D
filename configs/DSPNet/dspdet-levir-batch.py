default_scope = 'mmdet'
work_dir = '/mnt/dataset/yangqinzhe/mamba-pth/dspnet'
custom_imports = dict(imports=['mmdet.rsprompter'], allow_failed_imports=False)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

crop_size = (1024, 1024)
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

hf_sam_pretrain_name = "facebook/sam-vit-huge"
hf_sam_pretrain_ckpt_path = "pretrain_models/huggingface/hub/models--facebook--sam-vit-huge/snapshots/89080d6dcd9a900ebd712b13ff83ecf6f072e798/pytorch_model.bin"


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

# dataset settings
dataset_type = 'SSDDInsSegDataset'
code_root = '/mnt/home/xx/codes/RSPrompter'
data_root = '/mnt/dataset/yangqinzhe/Rsprompter-mamba/levir-ship/'

batch_size = 4
num_workers = 8
persistent_workers = True
indices = None

model = dict(
    type='DSPDet2D',
    data_preprocessor=data_preprocessor,
    backbone=dict(type='DSPBackbone2D', in_channels=3, max_channels=128, depth=34,  pool=False),
    head=dict(
        type='DSPHeadImage',
        in_channels=(64, 128, 128, 128),
        out_channels=128,
        n_reg_outs=6,
        n_classes=1,
        assigner=dict(
            type='DSPAssignerImage',
            top_pts_threshold=6,
        ),
        #volume_threshold=27,
        #r=7,
        prune_threshold=0.3,
        bbox_loss=dict(type='L1Loss', loss_weight=5.0)),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        indices=indices,
        data_root=data_root,
        ann_file='/mnt/dataset/yangqinzhe/Rsprompter-mamba/levir-ship/annotations/instances_train2017.json',
        data_prefix=dict(img='images/train2017/'),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
        indices=indices,
        data_root=data_root,
        ann_file='/mnt/dataset/yangqinzhe/Rsprompter-mamba/levir-ship/annotations/instances_val2017.json',
        data_prefix=dict(img='images/val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
    )
)
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        indices=indices,
        data_root=data_root,
        ann_file='/mnt/dataset/yangqinzhe/Rsprompter-mamba/levir-ship/annotations/instances_test2017.json',
        data_prefix=dict(img='images/test2017/'),
        test_mode=True,
        pipeline=test_pipeline,
    )
)

val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox'],
    format_only=False,
)
test_evaluator = val_evaluator

max_epochs = 800

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 0.0001

# learning rate




# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=batch_size)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=4, save_best='coco/bbox_mAP', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=True, interval=1, test_out_dir='vis_data')
)

vis_backends = [dict(type='LocalVisBackend'),
                #dict(type='WandbVisBackend', init_kwargs=dict(project='rsprompter-test', group='samseg-maskrcnn', name='samseg-maskrcnn-ssdd'))
                ]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

num_classes = 1

# # sam large model
# hf_sam_pretrain_name = "facebook/sam-vit-large"
# hf_sam_pretrain_ckpt_path = "~/.cache//huggingface/hub/models--facebook--sam-vit-large/snapshots/70009d56dac23ebb3265377257158b1d6ed4c802/pytorch_model.bin"
# # sam huge model
# hf_sam_pretrain_name = "facebook/sam-vit-huge"
# hf_sam_pretrain_ckpt_path = "~/.cache/huggingface/hub/models--facebook--sam-vit-huge/snapshots/89080d6dcd9a900ebd712b13ff83ecf6f072e798/pytorch_model.bin"


dataset_type = 'SSDDInsSegDataset'

#### should be changed align with your code root and data root
data_root = '/mnt/dataset/yangqinzhe/Rsprompter-mamba/SSDD'

batch_size=4
batch_size_per_gpu = 4
num_workers = 8
persistent_workers = True


find_unused_parameters = True






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

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05
    ),
    #accumulative_counts=2,
)
