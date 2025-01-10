_base_ = ['../../../../_base_/datasets/sam_dataset.py', '../../../../_base_/models/sam_mask_refinement.py']

data_root = 'UltraSAM_DATA/UltraSAM'

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img='MMOTU_2d/images'),
        ann_file='MMOTU_2d/annotations/test.agnostic.MMOTU_2d__coco.json',
    ),
)

test_evaluator = dict(
    ann_file='{}/MMOTU_2d/annotations/test.agnostic.MMOTU_2d__coco.json'.format(data_root),
)
