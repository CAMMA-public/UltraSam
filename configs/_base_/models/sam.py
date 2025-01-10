_base_ = '../default_runtime_iter.py'
custom_imports = dict(imports=['mmpretrain.models',
                               'endosam.datasets.transforms.custom_pipeline',
                               'endosam.datasets.transforms.point_formatting',
                               'endosam.visualization.point_visualization',
                               'endosam.models.detectors.SAM',
                               'endosam.models.backbones.vit_sam',
                               'endosam.models.backbones.MED_SA',
                               'endosam.models.dense_heads.sam_mask_decoder',
                               'endosam.models.dense_heads.sam_mask_class_decoder',
                               'endosam.datasets.evaluation.LabelMetric',
                               'endosam.models.utils.sam_layers',
                               'endosam.models.task_modules.assigners.SAMassigner',
                               'endosam.models.task_modules.prior_generators.prompt_encoder',
                               'endosam.models.task_modules.prior_generators.label_encoder',
                               'endosam.hooks.MonkeyPatchHook',
                               'endosam.hooks.FreezeHook',
                               'endosam.hooks.ValLossHook',
                               ], allow_failed_imports=False)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='PointVisualizer', vis_backends=vis_backends, name='visualizer')

# batch_augments = [
#     dict(
#         type='BatchFixedSizePad',
#         size=(1024, 1024),
#         img_pad_value=0,
#         pad_mask=True,
#         mask_pad_value=0,
#         pad_seg=False)
# ]
# data_preprocessor = dict(
#     type='DetDataPreprocessor',
#     # RGB format normalization parameters
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     # convert image from BGR to RGB
#     bgr_to_rgb=True,
#     pad_size_divisor=1024,
#     pad_mask=True,
#     mask_pad_value=0,
#     pad_seg=False,
#     batch_augments=batch_augments,
# )

data_preprocessor = dict(
    type='DetDataPreprocessor',
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    bgr_to_rgb=True,
    pad_size_divisor=1024,
)


load_from = 'weights/mapped_sam_vit_b_01ec64.pth'  # load mapped SAM VIT-B weights
model = dict(
    type='SAM',
    data_preprocessor=data_preprocessor,
    prompt_encoder=dict(
        type='SAMPaddingGenerator',
        label_encoder=dict(
            type='LabelEmbedEncoder',
            embed_dims=256,
        ),
    ),
    bbox_head=dict(
        type='SAMHead',
    ),
    backbone=dict(
        type='mmpretrain.ViTSAM',
        arch='base',
        img_size=1024,
        patch_size=16,
        out_channels=256,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
    ),
    decoder=dict(  # SAMTransformerDecoder
        num_layers=2,
        layer_cfg=dict(  # SAMTransformerLayer
            embedding_dim=256,
            num_heads=8,
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.1
            ),
        ),
    ),
    train_cfg=dict(
        assigner=dict(
            type='SAMassigner',
        )
    ),
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001,
                   weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=30000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=30000,
        by_epoch=False,
        milestones=[20000, 28888],
        gamma=0.1)
]

# custom_hooks = [dict(type="MonkeyPatchHook"), dict(type='ValLoss')]
custom_hooks = [dict(type="MonkeyPatchHook")]
