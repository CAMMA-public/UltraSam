_base_ = 'sam_dataset.py'

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args={{_base_.file_client_args}}),
    dict(type='FixScaleResize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='GetPointFromBox', number_of_points=[{{_base_.N_POINTS}}], test=True, normalize=False, get_center_point=True),
    dict(type='GetPointBox', test=True, normalize=False, max_jitter=0.0),
    dict(type='GetPromptType', prompt_probabilities=[0.0, 1.0]),
    dict(
        type='PackPointDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader
