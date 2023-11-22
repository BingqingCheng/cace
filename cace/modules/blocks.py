from typing import Callable, Union, Optional

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

__all__ = ["Dense", "AtomicEnergiesBlock"]


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: umber of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
            weight_init: weight initializer from current weight.
            bias_init: bias initializer from current bias.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y

class AtomicEnergiesBlock(nn.Module):
    def __init__(self, nz:int, trainable=True, atomic_energies: Optional[Union[np.ndarray, torch.Tensor]]=None):
        super().__init__()
        if atomic_energies is None:
            atomic_energies = torch.zeros(nz)
        else:
            assert len(atomic_energies.shape) == 1

        if trainable:
            self.atomic_energies = nn.Parameter(atomic_energies)
        else:
            self.register_buffer("atomic_energies", atomic_energies, torch.get_default_dtype())

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, self.atomic_energies)

    def __repr__(self):
        formatted_energies = ", ".join([f"{x:.4f}" for x in self.atomic_energies])
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"
