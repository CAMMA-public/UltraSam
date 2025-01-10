_base_ = ['../../../../../_base_/datasets/sam_dataset_classification.py', '../../../../../_base_/models/sam_classification.py']
data_root = 'UltraSAM_DATA/UltraSAM/datasets_classification/'

classes = ('lmym', 'gist')

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.VisionTransformer',
        out_type='avg_featmap',
        with_cls_token=False,
        arch='b',
        img_size=1024,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint='weights/vit-base-p16_pt-32xb128-mae_in1k_20220623-4c544545.pth')
    ),
    neck=None,
    head=dict(
        num_classes=len(classes),
        in_channels=768,
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
