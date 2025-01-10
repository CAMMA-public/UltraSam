import torch
from typing import Dict, Optional, Union
from mmengine.hooks import Hook
from mmengine.runner import Runner, autocast
from mmdet.registry import HOOKS


@HOOKS.register_module()
class ValLoss(Hook):
    """Save and print valid loss info
    Hacky & dirty
    """
    def __init__(self,
                 loss_list=[]) -> None:
        self.loss_list = loss_list

    def before_val(self, runner) -> None:
        # build the model
        self.model = runner.model
        self.loss_list = []

    def after_val_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """
            Figure every loss base self.loss_list and add the output information in logs.
        """
        if len(self.loss_list) > 0:
            loss_log = {}
            for lossInfo in self.loss_list:
                for tmp_loss_name, tmp_loss_value in lossInfo.items():
                    loss_log.setdefault(tmp_loss_name, []).append(tmp_loss_value)
            total_sum = 0  # Initialize the sum of scalars
            for loss_name, loss_values in loss_log.items():
                mean_loss = torch.mean(torch.stack(loss_values))
                runner.message_hub.update_scalar(f'val/{loss_name}_val', mean_loss)
                total_sum += mean_loss  # Add to the total sum

            # Update the sum scalar
            runner.message_hub.update_scalar('val/total_loss_val', total_sum)
        else:
            print('the model not support valid loss!')

    def after_val_iter(self,
                        runner: Runner,
                        batch_idx: int,
                        data_batch: Union[dict, tuple, list] = None,
                        outputs: Optional[dict] = None) -> None:
        """
        Figure the loss again
        Save all loss in self.loss_list.
        """
        with torch.no_grad():
            with autocast(enabled=runner.val_loop.fp16):
                data = self.model.data_preprocessor(data_batch, True)
                losses = self.model._run_forward(data, mode='loss')  # type: ignore
                self.loss_list.append(losses)
