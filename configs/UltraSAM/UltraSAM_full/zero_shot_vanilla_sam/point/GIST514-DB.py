_base_ = ['../../../../_base_/datasets/sam_dataset.py', '../../../../_base_/models/sam_mask_refinement.py']

data_root = 'UltraSAM_DATA/UltraSAM'

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='GIST514-DB/images'),
        ann_file='GIST514-DB/annotations/test.agnostic.GIST514-DB__coco.json',
    ),
)

test_evaluator = dict(
    ann_file='{}/GIST514-DB/annotations/test.agnostic.GIST514-DB__coco.json'.format(data_root),
)
