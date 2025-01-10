_base_ = ['../../_base_/datasets/sam_dataset_noMaskGT.py', '../../_base_/models/sam_mask_refinement.py']

data_root = 'UltraSAM_DATA/'

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img=''),
        ann_file='train.agnostic.noSmall.coco.json',
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img=''),
        ann_file='val.agnostic.noSmall.coco.json',
    ),
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img=''),
        ann_file='01-0012-D-J_TuBmod/coco_s1.json',
    ),
)

orig_val_evaluator = _base_.val_evaluator
orig_val_evaluator[0]['ann_file'] = '{}/detection_agnostic.json'.format(data_root)
val_evaluator = orig_val_evaluator

# orig_test_evaluator = _base_.test_evaluator
# orig_test_evaluator[0]['ann_file'] = '{}/merged_coco_test_agnostic.json'.format(data_root)
# test_evaluator = orig_test_evaluator

orig_test_evaluator = _base_.test_evaluator
orig_test_evaluator[0]['ann_file'] = '{}/01-0012-D-J_TuBmod/coco_s1.json'.format(data_root)
test_evaluator = orig_test_evaluator
