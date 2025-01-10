_base_ = ['../../../../../_base_/datasets/sam_dataset_segmentation.py', '../../../../../_base_/models/mask2former_sam.py']
data_root = 'UltraSAM_DATA/UltraSAM/'

classes = ('breast_nodule_benign', 'breast_nodule_malignant')

model = dict(
    backbone=dict(
        init_cfg=dict(prefix="backbone.", checkpoint="weights/mapped_sam_vit_b_01ec64.pth")
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
        data_prefix=dict(img='BUSBRA/images'),
        ann_file='BUSBRA/annotations/train.BUSBRA__coco.json',
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo={'classes': classes},
        data_prefix=dict(img='BUSBRA/images'),
        ann_file='BUSBRA/annotations/val.BUSBRA__coco.json',
    ),
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo={'classes': classes},
        data_prefix=dict(img='BUSBRA/images'),
        ann_file='BUSBRA/annotations/test.BUSBRA__coco.json',
    ),
)

orig_val_evaluator = _base_.val_evaluator
orig_val_evaluator['ann_file'] = '{}/BUSBRA/annotations/test.BUSBRA__coco.json'.format(data_root)
val_evaluator = orig_val_evaluator

orig_test_evaluator = _base_.test_evaluator
orig_test_evaluator['ann_file'] = '{}/BUSBRA/annotations/test.BUSBRA__coco.json'.format(data_root)
test_evaluator = orig_test_evaluator
