_base_ = ['../../../../../_base_/datasets/sam_dataset_segmentation.py', '../../../../../_base_/models/mask2former_sam.py']
data_root = 'UltraSAM_DATA/UltraSAM/'

classes = ('lmym', 'gist')

model = dict(
    backbone=dict(
        init_cfg=dict(prefix="backbone.", checkpoint="weights/mapped_medsam_vit_b.pth")
    ),
    panoptic_head=dict(
        num_things_classes=len(classes),
        loss_cls=dict(
            class_weight=[1.0] * len(classes) + [0.1]
        ),
    ),
    panoptic_fusion_head=dict(
        num_things_classes=len(classes),
    ),
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo={'classes': classes},
        data_prefix=dict(img='GIST514-DB/images'),
        ann_file='GIST514-DB/annotations/train.GIST514-DB__coco.json',
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo={'classes': classes},
        data_prefix=dict(img='GIST514-DB/images'),
        ann_file='GIST514-DB/annotations/val.GIST514-DB__coco.json',
    ),
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo={'classes': classes},
        data_prefix=dict(img='GIST514-DB/images'),
        ann_file='GIST514-DB/annotations/test.GIST514-DB__coco.json',
    ),
)

orig_val_evaluator = _base_.val_evaluator
orig_val_evaluator['ann_file'] = '{}/GIST514-DB/annotations/test.GIST514-DB__coco.json'.format(data_root)
val_evaluator = orig_val_evaluator

orig_test_evaluator = _base_.test_evaluator
orig_test_evaluator['ann_file'] = '{}/GIST514-DB/annotations/test.GIST514-DB__coco.json'.format(data_root)
test_evaluator = orig_test_evaluator
