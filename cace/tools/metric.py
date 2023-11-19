import torch

__all__ = ['compute_loss_metrics']

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
