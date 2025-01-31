_base_ = ['../../../../../_base_/datasets/sam_dataset_classification.py', '../../../../../_base_/models/sam_classification.py']
data_root = 'UltraSAM_DATA/UltraSAM/datasets_classification/'

classes = ('lmym', 'gist')

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='weights/resnet50-0676ba61.pth')
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        num_classes=len(classes),
        in_channels=2048,
    )
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        classes=classes,
        metainfo={'classes': classes},
        data_prefix='GIST514-DB',
        ann_file='GIST514-DB/annotations/train.txt',
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        classes=classes,
        metainfo={'classes': classes},
        data_prefix='GIST514-DB',
        ann_file='GIST514-DB/annotations/val_classification.txt',
    ),
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        classes=classes,
        metainfo={'classes': classes},
        data_prefix='GIST514-DB',
        ann_file='GIST514-DB/annotations/test_classification.txt',
    ),
)
