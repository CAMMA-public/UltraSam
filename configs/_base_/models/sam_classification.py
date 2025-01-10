_base_ = '../default_runtime_iter_classification.py'
custom_imports = dict(imports=['endosam.hooks.MonkeyPatchHook',
                               'endosam.models.backbones.vit_sam',
                               'endosam.models.backbones.MED_SA',
                               ], allow_failed_imports=False)


model = dict(
    type='mmpretrain.ImageClassifier',
    backbone=dict(
        type='mmpretrain.ViTSAM',
        arch='base',
        img_size=1024,
        patch_size=16,
        out_channels=0,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        out_type="avg_featmap",
        init_cfg=dict(type='Pretrained',
                      prefix="backbone.",
                      checkpoint="weights/mapped_sam_vit_b_01ec64.pth"
                      )
    ),
    head=dict(
        type='mmpretrain.MultiLabelLinearClsHead',
        num_classes=2,
        in_channels=768,
        topk=1,
        loss=dict(type='mmpretrain.FocalLoss', loss_weight=1.0),
    )
)

data_preprocessor = dict(
    type='mmpretrain.ClsDataPreprocessor',
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
    pad_size_divisor=1024,
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001,
                   weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

train_cfg = dict(
    type='mmpretrain.IterBasedTrainLoop', max_iters=2000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=2000,
        by_epoch=False,
        milestones=[1600],
        gamma=0.1)
]

custom_hooks = [dict(type="MonkeyPatchHook")]
