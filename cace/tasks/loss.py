from typing import Optional, Dict
import torch
import torch.nn as nn

from ..tools import compute_loss_metrics

__all__ = ["GetLoss"]

class GetLoss(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        predict_name: str,
        name: Optional[str] = None,
        target_name: Optional[str] = None,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Dict[str, list] = {"mae": [], "rmse": []},
    ):
        """
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
        """
        super().__init__()
        self.name = name or predict_name
        self.predict_name = predict_name
        self.target_name = target_name or predict_name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.train_metrics = metrics
        self.val_metrics = {k: [] for k, v in metrics.items()}
        self.test_metrics = {k: [] for k, v in metrics.items()}
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }

    def calculate_loss(self, 
                       pred: Dict[str, torch.Tensor], 
                       target: Optional[Dict[str, torch.Tensor]] = None,
                      ):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0
        
        if target is not None:
            loss = self.loss_weight * self.loss_fn(
                pred[self.predict_name], target[self.target_name]
            )
        elif self.predict_name != self.target_name:
            loss = self.loss_weight * self.loss_fn(
                pred[self.predict_name], pred[self.target_name]
            )
        else:
            raise ValueError("Target is None and predict_name is not equal to target_name")
        return loss

    def update_metrics(self, subset: str, 
                       pred: Dict[str, torch.Tensor], 
                       target: Optional[Dict[str, torch.Tensor]] = None,
                      ):
        for metric in self.metrics[subset].keys():
            if target is not None:
                value = compute_loss_metrics(metric, pred[self.predict_name], target[self.target_name])
            elif self.predict_name != self.target_name:
                value = compute_loss_metrics(metric, pred[self.predict_name], pred[self.target_name])
            else:
                raise ValueError("Target is None and predict_name is not equal to target_name")
            self.metrics[subset][metric].append(value)

    def clear_metric(self, subset: str):
        for metric in self.metrics[subset].keys():
            self.metrics[subset][metric] = []
