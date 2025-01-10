# dataset settings
dataset_type = 'CocoDataset'
file_client_args = dict(backend='disk')

image_size = (224, 224)
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        file_client_args=file_client_args),
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dummy_metainfo = {'classes': ('background', 'object',)}

train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=dummy_metainfo,
        data_prefix=dict(img='images/train/'),
        ann_file='annotations/train/annotation_coco.json',
        filter_cfg=dict(filter_empty_gt=True, min_size=16),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=dummy_metainfo,
        pipeline=test_pipeline,
        data_prefix=dict(img='images/val/'),
        ann_file='annotations/val/annotation_coco.json',
        filter_cfg=dict(filter_empty_gt=True, min_size=16),
        test_mode=False))
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='CocoMetric',
        metric=['bbox', 'segm'],
        classwise=True,
        format_only=False,
    ),
    # dict(
    #     type='SemSegMetric',
    #     iou_metrics=['mIoU', 'mDice', 'mFscore'],
    # ),
]
test_evaluator = val_evaluator
