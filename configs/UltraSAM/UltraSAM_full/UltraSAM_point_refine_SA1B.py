_base_ = ['../../_base_/datasets/sam_dataset.py', '../../_base_/models/sam_mask_refinement.py']

data_root = './UltraSAM_DATA/UltraSAM/'
# distributed=True
# log_file='tmp.log'
# log_level='DEBUG'

# model_wrapper_cfg = dict(
#     type='MMDistributedDataParallel',
#     find_unused_parameters=True,
#     detect_anomalous_params=True
# )

train_dataloader = dict(
    batch_size=8,
    # batch_size=1,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img=''),
        ann_file='train_with_SA1B.agnostic.noSmall.coco.json',
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
        data_prefix=dict(img='GIST514-DB/images'),
        ann_file='GIST514-DB/annotations/test.agnostic.GIST514-DB__coco.json',
    ),
)

# orig_val_evaluator = _base_.val_evaluator
# orig_val_evaluator[0]['ann_file'] = '{}/test.agnostic.noSmall.coco.json'.format(data_root)
# val_evaluator = orig_val_evaluator

# orig_test_evaluator = _base_.test_evaluator
# orig_test_evaluator[0]['ann_file'] = '{}/MMOTU_2d/annotations/test.agnostic.MMOTU_2d__coco.json'.format(data_root)
# test_evaluator = orig_test_evaluator


orig_val_evaluator = _base_.val_evaluator
orig_val_evaluator['ann_file'] = '{}/test.agnostic.noSmall.coco.json'.format(data_root)
val_evaluator = orig_val_evaluator

orig_test_evaluator = _base_.test_evaluator
orig_test_evaluator['ann_file'] = '{}/GIST514-DB/annotations/test.agnostic.GIST514-DB__coco.json'.format(data_root)
test_evaluator = orig_test_evaluator
