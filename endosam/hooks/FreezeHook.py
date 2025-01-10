from mmdet.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class FreezeHook(Hook):
    def __init__(
            self,
            freeze_all_but_lora: bool = False,
            freeze_all_but_adapter: bool = False):
        self.freeze_all_but_lora = freeze_all_but_lora
        self.freeze_all_but_adapter = freeze_all_but_adapter

    def after_load_checkpoint(self, runner, **kwargs):
        self.before_train(runner, **kwargs)

    def before_train(self, runner, **kwargs):
        model = runner.model
        # Handle DataParallel or DistributedDataParallel
        if hasattr(model, "module"):
            model = model.module  # Access the actual model if wrapped

        if self.freeze_all_but_lora:
            # Iterate over all parameters in the model
            for name, param in model.named_parameters():
                if '.lora_' in name:
                    param.requires_grad = True  # Keep LoRA parameters trainable
                else:
                    param.requires_grad = False  # Freeze all other parameters

        if self.freeze_all_but_adapter:
            # Iterate over all parameters in the model
            for name, param in model.named_parameters():
                if '_adapter' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
