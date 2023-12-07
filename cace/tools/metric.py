import logging
import torch
import torch.nn as nn
from typing import Optional, Dict

from .torch_tools import to_numpy

__all__ = ['Metrics', 'compute_loss_metrics']

def compute_loss_metrics(metric: str, y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Compute the loss metrics
    current options: mse, mae, rmse, r2
    """
    if metric == 'mse':
        return torch.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return torch.mean(torch.abs(y_true - y_pred))
    elif metric == 'rmse':
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    elif metric == 'r2':
        return 1 - torch.sum((y_true - y_pred) ** 2) / torch.sum((y_true - torch.mean(y_true)) ** 2)
    else:
        raise ValueError('Metric not implemented')

class Metrics(nn.Module):
    """
    Defines and calculate  metrics to be logged.
    """

    def __init__(
        self,
        target_name: str,
        predict_name: Optional[str] = None,
        name: Optional[str] = None,
        metrics: Dict[str, list] = {"mae": [], "rmse": []},
        per_atom: bool = False,
    ):
        """
        Args:
        target_name: name of the target in the dataset
        predict_name: name of the prediction in the model output
        name: name of the metrics
        metrics: dictionary of metrics to be calculated, mse, mae, rmse, r2
        per_atom: whether to calculate the metrics per atom
        """

        super().__init__()
        self.target_name = target_name
        self.predict_name = predict_name or target_name
        self.name = name or target_name

        self.per_atom = per_atom

        self.train_metrics = metrics
        self.val_metrics = {k: [] for k, v in metrics.items()}
        self.test_metrics = {k: [] for k, v in metrics.items()}
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }

    def update_metrics(self, subset: str, 
                       pred: Dict[str, torch.Tensor], 
                       target: Optional[Dict[str, torch.Tensor]] = None,
                      ):

        pred_tensor = pred[self.predict_name]
        if target is not None:
            target_tensor = target[self.target_name].detach()
        elif self.predict_name != self.target_name:
            target_tensor = pred[self.target_name].detach()
        else:
            raise ValueError("Target is None and predict_name is not equal to target_name")

        if self.per_atom:
            n_atoms = torch.bincount(target['batch'])
            pred_tensor = pred_tensor / n_atoms
            target_tensor = target_tensor / n_atoms

        for metric in self.metrics[subset].keys():
            value = compute_loss_metrics(metric, target_tensor, pred_tensor)
            self.metrics[subset][metric].append(value)

    def retrieve_metrics(self, subset: str):
        for metric_name, metric in self.metrics[subset].items():
            metric_now = to_numpy(torch.mean(torch.stack(metric))).item()
            print(
                f'{subset}_{self.name}_{metric_name}: {metric_now:.4f}',
            )
            logging.info(
                f'{subset}_{self.name}_{metric_name}: {metric_now:.4f}',
            )

    def clear_metrics(self, subset: str):
        for metric in self.metrics[subset].keys():
            self.metrics[subset][metric] = []
