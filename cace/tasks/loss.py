from typing import Optional, Dict
import torch
import torch.nn as nn

__all__ = ["GetLoss"]

class GetLoss(nn.Module):
    """
    Defines mappings to a loss function and weight for training
    """

    def __init__(
        self,
        predict_name: str,
        name: Optional[str] = None,
        target_name: Optional[str] = None,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
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
        self.predict_name = predict_name
        self.target_name = target_name or predict_name
        self.name = name or predict_name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight

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
