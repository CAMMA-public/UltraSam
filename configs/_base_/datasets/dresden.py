_base_ = './coco_instance.py'
custom_imports = dict(imports=['mmEus.datasets.transforms.transforms',
                               'mmEus.visualization.mmEus_visualizer',
                               'mmEus.datasets.transforms.formatting'], allow_failed_imports=False)

data_root='/DATA/DSAD'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('abdominal_wall', 'colon', 'inferior_mesenteric_artery', 'intestinal_veins',
           'liver', 'pancreas', 'small_intestine', 'spleen', 'stomach',
           'ureter', 'vesicular_glands')

file_client_args = dict(backend='disk')

visualizer = dict(
    type='DetPointVisualizer')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1280, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type="GetPointFromMask", number_of_points=3),
    dict(type='PackPointDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction')),
    # dict(type="printDebug"),
    # dict(type="debugStop"),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1280, 1024), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/ameyer/MMEUS/DATA/DSAD/DSAD_val.json',
    metric=['bbox', 'segm'],
    format_only=False)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/ameyer/MMEUS/DATA/DSAD/DSAD_test.json',
    metric=['bbox', 'segm'],
    format_only=False)


train_dataset=dict(
    type=dataset_type,
    # explicitly add your class names to the field `metainfo`
    metainfo=dict(classes=classes),
    pipeline=train_pipeline,
    ann_file='DSAD_train.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=0),
    data_prefix=dict(img=''),
    data_root=data_root,
    
)

val_dataset=dict(
    type=dataset_type,
    # explicitly add your class names to the field `metainfo`
    metainfo=dict(classes=classes),
    test_mode=True,
    pipeline=test_pipeline,
    ann_file='DSAD_val.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=0),
    data_prefix=dict(img=''),
    data_root=data_root,
)
test_dataset=dict(
    type=dataset_type,
    # explicitly add your class names to the field `metainfo`
    metainfo=dict(classes=classes),
    test_mode=True,
    pipeline=test_pipeline,
    ann_file='DSAD_test.json',
    filter_cfg=dict(filter_empty_gt=False, min_size=0),
    data_prefix=dict(img=''),
    data_root=data_root,
)

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=train_dataset
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=val_dataset
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=test_dataset
)