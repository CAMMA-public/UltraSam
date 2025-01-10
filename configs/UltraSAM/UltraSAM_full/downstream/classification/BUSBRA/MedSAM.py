_base_ = ['../../../../../_base_/datasets/sam_dataset_classification.py', '../../../../../_base_/models/sam_classification.py']
data_root = 'UltraSAM_DATA/UltraSAM/datasets_classification/'

classes = ('breast_nodule_benign', 'breast_nodule_malignant')

model = dict(
    backbone=dict(
        init_cfg=dict(checkpoint="weights/mapped_medsam_vit_b.pth")
    ),
    head=dict(
        num_classes=len(classes),
    ),
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        classes=classes,
        metainfo={'classes': classes},
        data_prefix='BUSBRA',
        ann_file='BUSBRA/annotations/train.txt',
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        classes=classes,
        metainfo={'classes': classes},
        data_prefix='BUSBRA',
        ann_file='BUSBRA/annotations/val_classification.txt',
    ),
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        classes=classes,
        metainfo={'classes': classes},
        data_prefix='BUSBRA',
        ann_file='BUSBRA/annotations/test_classification.txt',
    ),
)
