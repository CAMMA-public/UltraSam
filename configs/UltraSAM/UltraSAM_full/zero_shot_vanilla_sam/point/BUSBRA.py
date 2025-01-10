_base_ = ['../../../../_base_/datasets/sam_dataset.py', '../../../../_base_/models/sam_mask_refinement.py']

data_root = 'UltraSAM_DATA'

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='BUSBRA/images'),
        ann_file='BUSBRA/annotations/test.agnostic.BUSBRA__coco.json',
    ),
)

test_evaluator = dict(
    ann_file='{}/BUSBRA/annotations/test.agnostic.BUSBRA__coco.json'.format(data_root),
)
