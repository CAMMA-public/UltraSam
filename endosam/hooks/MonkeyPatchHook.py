from mmdet.registry import HOOKS
from mmengine.hooks import Hook
from endosam.models.utils.custom_functional import multi_head_attention_forward
import torch.nn.functional as F

@HOOKS.register_module()
class MonkeyPatchHook(Hook):
    def before_run(self, runner) -> None:
        F.multi_head_attention_forward = multi_head_attention_forward
