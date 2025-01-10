_base_ = '../UltraSAM_full/UltraSAM_box_refine.py'

data_root = 'UltraSAM_DATA'

train_dataloader = dict(
    dataset=dict(
        ann_file='train.noStructure.GIST514-DB.json',
    ),
)

test_ann_file = 'GIST514-DB/annotations/test.agnostic.GIST514-DB__coco.json'
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img='GIST514-DB/images/'),
        ann_file=test_ann_file,
    ),
)
test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img='GIST514-DB/images/'),
        ann_file=test_ann_file,
    ),
)

orig_val_evaluator = _base_.val_evaluator
orig_val_evaluator[0]['ann_file'] = '{}/{}'.format(data_root, test_ann_file)
val_evaluator = orig_val_evaluator

orig_test_evaluator = _base_.test_evaluator
orig_test_evaluator[0]['ann_file'] = '{}/{}'.format(data_root, test_ann_file)
test_evaluator = orig_test_evaluator
