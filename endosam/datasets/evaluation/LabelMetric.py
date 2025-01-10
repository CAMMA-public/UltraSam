from mmpretrain.evaluation.metrics.multi_label import MultiLabelMetric
from mmpretrain.evaluation.metrics.single_label import SingleLabelMetric, _precision_recall_f1_support, to_tensor
from mmpretrain.structures import label_to_onehot
from typing import List, Optional, Sequence, Union
import torch.nn.functional as F
from mmdet.registry import METRICS
import torch
import numpy as np

@METRICS.register_module()
class SAMLabelMetric(SingleLabelMetric):
    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # shape N, C
            pred_score = data_sample['pred_instances']['pred_score'].cpu()
            # shape N
            gt_label = data_sample['pred_instances']['labels'].cpu()

            # Check if we need to iterate (when N > 1)
            if gt_label.shape[0] > 1:
                # TODO take mean?
                for i, (score, label) in enumerate(zip(pred_score, gt_label)):
                    # Create a new dictionary for each element
                    result = dict()
                    result['pred_score'] = score[None]
                    result['gt_label'] = label[None]
                    # Append each result to self.results
                    # print(result)
                    self.results.append(result)
            else:
                # If there's only one element, create a single result dict
                result = dict()
                result['pred_score'] = pred_score
                result['gt_label'] = gt_label
                # print(result)
                # Append the single result to self.results
                self.results.append(result)

            # print(pred_score.shape, gt_label.shape)

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
        metrics = {}    

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

        # concat
        # 
        # breakpoint()
        target = torch.cat([res['gt_label'] for res in results])
        if 'pred_score' in results[0]:
            pred = torch.stack([res['pred_score'] for res in results])
            metrics_list = self.calculate(
                pred, target, thrs=self.thrs, average=self.average)

            multi_thrs = len(self.thrs) > 1
            for i, thr in enumerate(self.thrs):
                if multi_thrs:
                    suffix = '_no-thr' if thr is None else f'_thr-{thr:.2f}'
                else:
                    suffix = ''

                for k, v in pack_results(*metrics_list[i]).items():
                    metrics[k + suffix] = v
        else:
            # If only label in the `pred_label`.
            pred = torch.cat([res['pred_label'] for res in results])
            res = self.calculate(
                pred,
                target,
                average=self.average,
                num_classes=results[0]['num_classes'])
            metrics = pack_results(*res)

        result_metrics = dict()
        for k, v in metrics.items():

            if self.average is None:
                result_metrics[k + '_classwise'] = v.cpu().detach().tolist()
            elif self.average == 'micro':
                result_metrics[k + f'_{self.average}'] = v.item()
            else:
                result_metrics[k] = v.item()

        return result_metrics


    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        thrs: Sequence[Union[float, None]] = (0., ),
        average: Optional[str] = 'macro',
        num_classes: Optional[int] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate the precision, recall, f1-score and support.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
            thrs (Sequence[float | None]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. None means no thresholds.
                Defaults to (0., ).
            average (str | None): How to calculate the final metrics from
                the confusion matrix of every category. It supports three
                modes:

                - `"macro"`: Calculate metrics for each category, and calculate
                  the mean value over all categories.
                - `"micro"`: Average the confusion matrix over all categories
                  and calculate metrics on the mean confusion matrix.
                - `None`: Calculate metrics of every category and output
                  directly.

                Defaults to "macro".
            num_classes (Optional, int): The number of classes. If the ``pred``
                is label instead of scores, this argument is required.
                Defaults to None.

        Returns:
            Tuple: The tuple contains precision, recall and f1-score.
            And the type of each item is:

            - torch.Tensor: If the ``pred`` is a sequence of label instead of
              score (number of dimensions is 1). Only returns a tensor for
              each metric. The shape is (1, ) if ``classwise`` is False, and
              (C, ) if ``classwise`` is True.
            - List[torch.Tensor]: If the ``pred`` is a sequence of score
              (number of dimensions is 2). Return the metrics on each ``thrs``.
              The shape of tensor is (1, ) if ``classwise`` is False, and (C, )
              if ``classwise`` is True.
        """
        average_options = ['micro', 'macro', None]
        assert average in average_options, 'Invalid `average` argument, ' \
            f'please specify from {average_options}.'

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            assert num_classes is not None, \
                'Please specify the `num_classes` if the `pred` is labels ' \
                'intead of scores.'
            gt_positive = F.one_hot(target.flatten(), num_classes)
            pred_positive = F.one_hot(pred.to(torch.int64), num_classes)
            return _precision_recall_f1_support(pred_positive, gt_positive,
                                                average)
        else:
            # For pred score, calculate on all thresholds.
            num_classes = pred.size(2)
            pred_score, pred_label = torch.topk(pred, k=1)
            pred_score = pred_score.flatten()
            pred_label = pred_label.flatten()

            # breakpoint()

            gt_positive = F.one_hot(target.flatten(), num_classes)

            results = []
            for thr in thrs:
                pred_positive = F.one_hot(pred_label, num_classes)
                if thr is not None:
                    pred_positive[pred_score <= thr] = 0
                results.append(
                    _precision_recall_f1_support(pred_positive, gt_positive,
                                                 average))

            return results
    # def process(self, data_batch, data_samples: Sequence[dict]):
    #     """Process one batch of data samples.

    #     The processed results should be stored in ``self.results``, which will
    #     be used to computed the metrics when all batches have been processed.

    #     Args:
    #         data_batch: A batch of data from the dataloader.
    #         data_samples (Sequence[dict]): A batch of outputs from the model.
    #     """
    #     for data_sample in data_samples:
    #         # breakpoint()
    #         # data_sample['pred_instances']['pred_score']
    #         pred_score = data_sample['pred_instances']['pred_score'].clone()
    #         num_classes = pred_score.size()[-1]
    #         gt_labels = data_sample['pred_instances']['labels']

    #         # if pred_score.shape[0] > 1:
    #         #     for pred_scr, gt_scr in zip(pred_score, gt_labels):
    #         #         # result = dict()
    #         #         # result['pred_score'] = pred_scr
    #         #         # result['gt_score'] = label_to_onehot(gt_scr, num_classes)
    #         #         self.results.append(
    #         #             dict(
    #         #                 pred_score=pred_scr[None],
    #         #                 gt_score=label_to_onehot(gt_scr, num_classes)
    #         #             )
    #         #         )
    #         # else:
    #         result = dict()
    #         result['pred_score'] = pred_score
    #         result['gt_score'] = label_to_onehot(gt_labels,
    #                                                 num_classes)
    #         self.results.append(result)

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
    #     # metrics = {}

    #     # target = [res['gt_score'] for res in results]
    #     # pred = [res['pred_score'] for res in results]

    #     metrics = {}
    #     pred = []
    #     target = []
    #     for res in results:
    #         if res['pred_score'].shape[0] > 1:
    #             for pred_scr, gt_scr in zip(res['pred_score'], res['gt_score']):
    #                 print("====", pred_scr[None], gt_scr[None])
    #                 pred.append(pred_scr[None])
    #                 target.append(gt_scr[None])
    #         else:
    #             print(res['pred_score'].shape, res['gt_score'].shape)
    #             pred.append(res['pred_score'])
    #             target.append(res['gt_score'])

    #     metric_res = self.calculate(
    #         pred,
    #         target,
    #         pred_indices=True,
    #         target_indices=True,
    #         num_classes=results[0]['pred_score'].size()[-1],
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
