from typing import Optional, Dict, Union, Callable
import torch
import torch.nn as nn

__all__ = ["GetLoss", "GetRegularizationLoss", "GetVarianceLoss"]

class GetLoss(nn.Module):
    """
    Defines mappings to a loss function and weight for training
    """

    def __init__(
        self,
        target_name: str,
        predict_name: Optional[str] = None,
        output_index: Optional[int] = None, # only used for multi-output models
        name: Optional[str] = None,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: Union[float, Callable] = 1.0, # Union[float, Callable] means that the type can be either float or callable
    ):
        """
        Args:
            target_name: Name of target in training batch.
            name: name of the loss object
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
                         This can be a float or a callable that takes in the loss_weight_args
                         For example, if we want the loss weight to be dependent on the epoch number
                         if training == True and a default value of 1.0 otherwise,
                         loss_weight can be, e.g., lambda training, epoch: 1.0 if not training else epoch / 100
        """
        super().__init__()
        self.target_name = target_name
        self.predict_name = predict_name or target_name
        self.output_index = output_index
        self.name = name or target_name
        self.loss_fn = loss_fn
        # the loss_weight can either be a float or a callable        
        self.loss_weight = loss_weight

    def forward(self, 
                pred: Dict[str, torch.Tensor], 
                target: Optional[Dict[str, torch.Tensor]] = None,
                loss_args: Optional[Dict[str, torch.Tensor]] = None
                ):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        if isinstance(self.loss_weight, Callable):
            if loss_args is None:
                loss_weight = self.loss_weight()
            else:
                loss_weight = self.loss_weight(**loss_args)
        else: 
            loss_weight = self.loss_weight

        if target is not None:
            loss = loss_weight * self.loss_fn(
                pred[self.predict_name] if self.output_index is None else pred[self.predict_name][..., self.output_index], 
                target[self.target_name]
            )
        elif self.predict_name != self.target_name:
            loss = loss_weight * self.loss_fn(
                pred[self.predict_name] if self.output_index is None else pred[self.predict_name][..., self.output_index], 	
                pred[self.target_name]
            )
        else:
            raise ValueError("Target is None and predict_name is not equal to target_name")
        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name}, loss_fn={self.loss_fn}, loss_weight={self.loss_weight})"
            )

class GetRegularizationLoss(nn.Module):
    def __init__(self, loss_weight, model):
        super().__init__()
        self.loss_weight = loss_weight 
        self.model = model

    def forward(self,
                *args,
                ):
        regularization_loss = 0.0
        for param in self.model.parameters():
            if param.requires_grad:
                regularization_loss += torch.norm(param, p=2)
        regularization_loss *= self.loss_weight
        return regularization_loss

class GetVarianceLoss(nn.Module):
    def __init__(
        self,
        target_name: str,
        loss_weight: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.target_name = target_name
        self.loss_weight = loss_weight
        self.name = name

    def forward(self, pred: Dict[str, torch.Tensor], *args):

        # Compute the variance along the first dimension (across different entries)
        variances = torch.var(pred[self.target_name], dim=0)

        # Calculate the mean of the variances
        mean_variance = torch.mean(variances)
        mean_variance = mean_variance * self.loss_weight

        return mean_variance
