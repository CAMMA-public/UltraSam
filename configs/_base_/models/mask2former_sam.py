_base_ = '../default_runtime_iter.py'
custom_imports = dict(imports=['mmpretrain.models',
                               'endosam.datasets.transforms.custom_pipeline',
                               'endosam.datasets.transforms.point_formatting',
                               'endosam.visualization.point_visualization',
                               'endosam.models.detectors.SAM',
                               'endosam.models.backbones.endoViT',
                               'endosam.models.dense_heads.sam_mask_decoder',
                               'endosam.models.dense_heads.sam_mask_class_decoder',
                               'endosam.datasets.evaluation.LabelMetric',
                               'endosam.models.utils.sam_layers',
                               'endosam.models.task_modules.prior_generators.prompt_encoder',
                               'endosam.models.task_modules.prior_generators.label_encoder',
                               'endosam.hooks.MonkeyPatchHook',
                               ], allow_failed_imports=False)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='PointVisualizer', vis_backends=vis_backends, name='visualizer')

# preprocessing
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=(1024, 1024),
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    batch_augments=batch_augments,
)


num_things_classes = 80
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='Mask2Former',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.ViTSAM',
        arch='base',
        img_size=1024,
        patch_size=16,
        out_channels=0,
        out_indices=(2, 5, 8, 11),  # each stage a shape (B, C, H, W), C expected last?
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        init_cfg=dict(type='Pretrained', checkpoint="weights/UltraSam.pth"),
    ),
    panoptic_head=dict(
        type='Mask2FormerHead',
        init_cfg=dict(type='Pretrained', checkpoint="weights/mask2former_r50_coco.pth", prefix="panoptic_head."),
        # in_channels=[256, 256, 256, 256],  # pass to pixel_decoder inside
        in_channels=[768, 768, 768, 768],  # pass to pixel_decoder inside
        strides=[16, 16, 16, 16],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=dict(type='Pretrained', checkpoint="weights/mask2former_r50_coco.pth", prefix="panoptic_head.transformer_decoder."),),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=False,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
    init_cfg=None)

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
paramwise_cfg = dict(
    custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        'query_embed': embed_multi,
        'query_feat': embed_multi,
        'level_embed': embed_multi,
    },
    norm_decay_mult=0.0,
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001,
                   weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=paramwise_cfg,
)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=30000, val_interval=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=8000,
        by_epoch=False,
        milestones=[6000],
        gamma=0.1)
]

custom_hooks = [dict(type="MonkeyPatchHook")]
