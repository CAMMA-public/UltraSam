# dataset settings
dataset_type = 'CocoDataset'
file_client_args = dict(backend='disk')
backend_args = None

N_POINTS = 1  # number of prompt sampled per instances
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FixScaleResize', scale=(1024, 1024), keep_ratio=True),
    dict(type='GetPointFromMask', number_of_points=[N_POINTS], test=False, normalize=False),
    dict(type='GetPointBox', test=False, normalize=False),
    dict(type='GetPromptType', prompt_probabilities=[0.5, 0.5]),
    dict(type='PackPointDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='FixScaleResize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='GetPointFromMask', number_of_points=[N_POINTS], test=True, normalize=False, get_center_point=True),
    dict(type='GetPointBox', test=True, normalize=False),
    dict(type='GetPromptType', prompt_probabilities=[1.0, 0.0]),
    dict(
        type='PackPointDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dummy_metainfo = {'classes': ('background', 'tumor',)}
dummy_metainfo = {'classes': ('object',)}

train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=dummy_metainfo,
        data_prefix=dict(img='images/train/'),
        ann_file='annotations/train/annotation_coco.json',
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        # indices=200,
        metainfo=dummy_metainfo,
        pipeline=test_pipeline,
        data_prefix=dict(img='images/val/'),
        ann_file='annotations/val/annotation_coco.json',
        # filter_cfg=dict(filter_empty_gt=True),
        test_mode=True,
        backend_args=backend_args)
    )
test_dataloader = val_dataloader

# val_evaluator = [
#     dict(
#         type='CocoMetric',
#         metric=['bbox', 'segm'],
#         format_only=False,
#         classwise=True),
#     # dict(
#     #     type='SemSegMetric',
#     #     iou_metrics=['mIoU', 'mDice', 'mFscore'],
#     #     collect_device='gpu',),
# ]

val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    format_only=False,
    classwise=True,
    backend_args=backend_args,
    # use_mp_eval=True
)

test_evaluator = val_evaluator
