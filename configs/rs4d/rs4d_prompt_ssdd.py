_base_ = ['_base_/rs4d_anchor.py']

work_dir = './work_dirs/rs4d/rs4d_ssdd'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=4, save_best='coco/bbox_mAP', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='DetVisualizationHook', draw=True, interval=1, test_out_dir='vis_data')
)

vis_backends = [dict(type='LocalVisBackend'),
                #dict(type='WandbVisBackend', init_kwargs=dict(project='rs4d-test', group='rs4d-prompt', name=work_dir.split('/')[-1]))
                ]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

num_classes = 1
prompt_shape = (70, 5)  # (per img pointset, per pointset point)

#### should be changed when using different pretrain model

# sam base model
hf_sam_pretrain_name = "work_dirs/sam_cache/sam_vit_base"
# huggingface model name, e.g. facebook/sam-vit-base
# or local repo path, e.g. work_dirs/sam_cache/sam_vit_base
hf_sam_pretrain_ckpt_path = "work_dirs/sam_cache/sam_vit_base/pytorch_model.bin"
# # sam large model
# hf_sam_pretrain_name = "facebook/sam-vit-large"
# hf_sam_pretrain_ckpt_path = "~/.cache//huggingface/hub/models--facebook--sam-vit-large/snapshots/70009d56dac23ebb3265377257158b1d6ed4c802/pytorch_model.bin"
# # sam huge model
# hf_sam_pretrain_name = "facebook/sam-vit-huge"
# hf_sam_pretrain_ckpt_path = "~/.cache/huggingface/hub/models--facebook--sam-vit-huge/snapshots/89080d6dcd9a900ebd712b13ff83ecf6f072e798/pytorch_model.bin"


model = dict(
    decoder_freeze=False,
    backbone_freeze=False,
    shared_image_embedding=dict(
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
    ),
    backbone=dict(
        _delete_=True,
        type='RSMamba',
        arch='base',
        out_type='featmap_hwc',
        out_indices=list(range(0, 24)),
    ),
    neck=dict(
        type='RSFPN',
        feature_aggregator=dict(
            type='RSFeatureAggregator',
            in_channels='rsmamba_base',
            out_channels=256,
            hidden_channels=32,
            select_layers=range(2, 24, 2),
        ),
        feature_spliter=dict(
            type='RSSimpleFPN',
            backbone_channel=256,
            in_channels=[64, 128, 256, 256],
            out_channels=256,
            num_outs=5,
            norm_cfg=dict(type='LN2d', requires_grad=True)),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=num_classes,
        ),
        mask_head=dict(
            mask_decoder=dict(
                hf_pretrain_name=hf_sam_pretrain_name,
                init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)
            ),
            per_pointset_point=prompt_shape[1],
            with_sincos=False,
        ),
    ),
)

dataset_type = 'SSDDInsSegDataset'

#### should be changed align with your code root and data root
code_root = '/root/autodl-tmp/project/project_code/workspace_pytorch/rs4d'
data_root = '/root/autodl-tmp/project/project_code/workspace_pytorch/rs4d/data/SSDD'

batch_size=4
batch_size_per_gpu = 4
num_workers = 8
persistent_workers = True
train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=code_root + '/data/SSDD/annotations/SSDD_instances_train.json',
        data_prefix=dict(img='imgs'),
    )
)

val_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=code_root + '/data/SSDD/annotations/SSDD_instances_val.json',
        data_prefix=dict(img='imgs'),
    )
)

find_unused_parameters = True

test_dataloader = val_dataloader
resume = False
load_from = None

base_lr = 0.0001
max_epochs = 800

train_cfg = dict(max_epochs=max_epochs)

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

#### AMP training config
runner_type = 'Runner'
# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     dtype='float16',
#     optimizer=dict(
#         type='AdamW',
#         lr=base_lr,
#         weight_decay=0.05),
#     accumulative_counts=8,
# )
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05),
    #accumulative_counts=2,
)

# ### DeepSpeed training config
# runner_type = 'FlexibleRunner'
# strategy = dict(
#     type='DeepSpeedStrategy',
#     fp16=dict(
#         enabled=True,
#         auto_cast=False,
#         fp16_master_weights_and_grads=False,
#         loss_scale=0,
#         loss_scale_window=500,
#         hysteresis=2,
#         min_loss_scale=1,
#         initial_scale_power=15,
#     ),
#     inputs_to_half=['inputs'],
#     zero_optimization=dict(
#         stage=2,
#         allgather_partitions=True,
#         allgather_bucket_size=2e8,
#         reduce_scatter=True,
#         reduce_bucket_size='auto',
#         overlap_comm=True,
#         contiguous_gradients=True,
#     ),
# )
# optim_wrapper = dict(
#     type='DeepSpeedOptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=base_lr,
#         weight_decay=0.05
#     )
# )
