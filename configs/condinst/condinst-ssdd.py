_base_ = '../common/ms-poly-90k_coco-instance.py'
work_dir = '/mnt/dataset/yangqinzhe/Rsprompter-mamba-pth/new/work_dirs/rsprompter-test/condinst-ssdd'
default_scope = 'mmdet'
custom_imports = dict(imports=['mmdet.rsprompter'], allow_failed_imports=False)
# model settings
model = dict(
    type='CondInst',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='EfficientNetB0Backbone',
        out_indices=(1, 2, 3, 4),  # 对应 [24, 40, 112, 320]
        pretrained=False,  # 不用 timm 自己下
        checkpoint_path='/mnt/user/yangqinzhe/project/project_code/RSPrompter-mamba/pytorch_model.bin',
        in_chans=3,
    ),
    neck=dict(
        type='FPN',
        in_channels=[24, 40, 112, 320],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CondInstBboxHead',
        num_params=169,
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    mask_head=dict(
        type='CondInstMaskHead',
        num_layers=3,
        feat_channels=8,
        size_of_interest=8,
        mask_out_stride=4,
        max_masks_to_train=300,
        mask_feature_head=dict(
            in_channels=256,
            feat_channels=128,
            start_level=0,
            end_level=2,
            out_channels=8,
            mask_stride=8,
            num_stacked_convs=4,
            norm_cfg=dict(type='BN', requires_grad=True)),
        loss_mask=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            eps=5e-6,
            loss_weight=1.0)),
    # model training and testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr=0.5))

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.01))
