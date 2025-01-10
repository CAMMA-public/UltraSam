_base_ = './sam_classification.py'

model = dict(
    backbone=dict(
        type='MED_SA',
    ),
)
