# dataset settings
dataset_type = 'mmpretrain.CustomDataset'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='mmpretrain.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FixScaleResize', scale=(1024, 1024), keep_ratio=True),
    dict(type="Pad", size=(1024, 1024)),
    dict(type='mmpretrain.PackInputs')
]

test_pipeline = [
    dict(type='mmpretrain.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='FixScaleResize', scale=(1024, 1024), keep_ratio=True),
    dict(type="Pad", size=(1024, 1024)),
    dict(type='mmpretrain.PackInputs'),
]

dummy_metainfo = {'classes': ('object')}

train_dataloader = dict(
    batch_size=8,
    num_workers=6,
    persistent_workers=True,
    # sampler=dict(type='mmpretrain.DefaultSampler', shuffle=True),
    sampler=dict(type='InfiniteSampler', shuffle=True),
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(img='images/train/'),
        with_label=True,  # or False for unsupervised tasks
        classes=['A', 'B',],  # The name of every category.
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='mmpretrain.DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(img='images/val/'),
        with_label=True,  # or False for unsupervised tasks
        classes=['A', 'B',],  # The name of every category.
        pipeline=test_pipeline)
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmpretrain.SingleLabelMetric',
)
test_evaluator = val_evaluator
