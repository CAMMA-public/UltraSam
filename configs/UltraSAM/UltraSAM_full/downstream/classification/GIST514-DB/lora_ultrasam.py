_base_ = ['../../../../../_base_/datasets/sam_dataset_classification.py', '../../../../../_base_/models/sam_classification_lora.py']
data_root = 'UltraSAM_DATA/UltraSAM/datasets_classification/'

classes = ('lmym', 'gist')

model = dict(
    backbone=dict(
        module=dict(
            init_cfg=dict(
                checkpoint="weights/UltraSam.pth"
            ),
        ),
    ),
    head=dict(
        num_classes=len(classes)
    ),
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