_base_ = './sam_mask_refinement.py'

model = dict(
    backbone=dict(
        type='MED_SA',
    ),
)

custom_hooks = [dict(type="MonkeyPatchHook"), dict(type="FreezeHook", freeze_all_but_adapter=True)]
