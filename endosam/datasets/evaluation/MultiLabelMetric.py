from mmpretrain.evaluation.metrics.multi_label import MultiLabelMetric
from mmpretrain.structures import label_to_onehot
from typing import List, Optional, Sequence, Union
from mmdet.registry import METRICS
import torch

@METRICS.register_module()
class SAMMultiLabelMetric(MultiLabelMetric):
    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # breakpoint()
            # data_sample['pred_instances']['pred_score']
            pred_score = data_sample['pred_instances']['pred_score'].clone()
            num_classes = pred_score.size()[-1]
            gt_labels = data_sample['pred_instances']['labels']

            # if pred_score.shape[0] > 1:
            #     for pred_scr, gt_scr in zip(pred_score, gt_labels):
            #         # result = dict()
            #         # result['pred_score'] = pred_scr
            #         # result['gt_score'] = label_to_onehot(gt_scr, num_classes)
            #         self.results.append(
            #             dict(
            #                 pred_score=pred_scr[None],
            #                 gt_score=label_to_onehot(gt_scr, num_classes)
            #             )
            #         )
            # else:
            result = dict()
            result['pred_score'] = pred_score
            result['gt_score'] = label_to_onehot(gt_labels,
                                                    num_classes)
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.
        # metrics = {}

        # target = [res['gt_score'] for res in results]
        # pred = [res['pred_score'] for res in results]

        metrics = {}
        pred = []
        target = []
        for res in results:
            if res['pred_score'].shape[0] > 1:
                for pred_scr, gt_scr in zip(res['pred_score'], res['gt_score']):
                    print("====", pred_scr[None], gt_scr[None])
                    pred.append(pred_scr[None])
                    target.append(gt_scr[None])
            else:
                print(res['pred_score'].shape, res['gt_score'].shape)
                pred.append(res['pred_score'])
                target.append(res['gt_score'])

        metric_res = self.calculate(
            pred,
            target,
            pred_indices=True,
            target_indices=True,
            num_classes=results[0]['pred_score'].size()[-1],
            average=self.average,
            thr=self.thr,
            topk=self.topk)

        def pack_results(precision, recall, f1_score, support):
            single_metrics = {}
            if 'precision' in self.items:
                single_metrics['precision'] = precision
            if 'recall' in self.items:
                single_metrics['recall'] = recall
            if 'f1-score' in self.items:
                single_metrics['f1-score'] = f1_score
            if 'support' in self.items:
                single_metrics['support'] = support
            return single_metrics

        if self.thr:
            suffix = '' if self.thr == 0.5 else f'_thr-{self.thr:.2f}'
            for k, v in pack_results(*metric_res).items():
                metrics[k + suffix] = v
        else:
            for k, v in pack_results(*metric_res).items():
                metrics[k + f'_top{self.topk}'] = v

        result_metrics = dict()
        for k, v in metrics.items():
            if self.average is None:
                result_metrics[k + '_classwise'] = v.detach().cpu().tolist()
            elif self.average == 'macro':
                result_metrics[k] = v.item()
            else:
                result_metrics[k + f'_{self.average}'] = v.item()
        return result_metrics

    # def compute_metrics(self, results: List):
    #     """Compute the metrics from processed results.

    #     Args:
    #         results (list): The processed results of each batch.

    #     Returns:
    #         Dict: The computed metrics. The keys are the names of the metrics,
    #         and the values are corresponding results.
    #     """
    #     # NOTICE: don't access `self.results` from the method. `self.results`
    #     # are a list of results from multiple batch, while the input `results`
    #     # are the collected results.
    #     metrics = {}
    #     pred = []
    #     target = []
    #     for res in results:
    #         if res['pred_score'].shape[0] > 1:
    #             for pred_scr, gt_scr in zip(res['pred_score'], res['gt_score']):
    #                 breakpoint()
    #                 pred.append(pred_scr[None])
    #                 target.append(gt_scr[None])
    #         else:
    #             print(res['pred_score'].shape, res['gt_score'].shape)
    #             pred.append(res['pred_score'])
    #             target.append(res['gt_score'][None])
    #     # breakpoint()
    #     # target = torch.stack([res['gt_score'] for res in results])
    #     # pred = torch.stack([res['pred_score'] for res in results])
    #     target = torch.stack(target)
    #     pred = torch.stack(pred)

    #     metric_res = self.calculate(
    #         pred,
    #         target,
    #         pred_indices=False,
    #         target_indices=False,
    #         average=self.average,
    #         thr=self.thr,
    #         topk=self.topk)

    #     def pack_results(precision, recall, f1_score, support):
    #         single_metrics = {}
    #         if 'precision' in self.items:
    #             single_metrics['precision'] = precision
    #         if 'recall' in self.items:
    #             single_metrics['recall'] = recall
    #         if 'f1-score' in self.items:
    #             single_metrics['f1-score'] = f1_score
    #         if 'support' in self.items:
    #             single_metrics['support'] = support
    #         return single_metrics

    #     if self.thr:
    #         suffix = '' if self.thr == 0.5 else f'_thr-{self.thr:.2f}'
    #         for k, v in pack_results(*metric_res).items():
    #             metrics[k + suffix] = v
    #     else:
    #         for k, v in pack_results(*metric_res).items():
    #             metrics[k + f'_top{self.topk}'] = v

    #     result_metrics = dict()
    #     for k, v in metrics.items():
    #         if self.average is None:
    #             result_metrics[k + '_classwise'] = v.detach().cpu().tolist()
    #         elif self.average == 'macro':
    #             result_metrics[k] = v.item()
    #         else:
    #             result_metrics[k + f'_{self.average}'] = v.item()
    #     return result_metrics
