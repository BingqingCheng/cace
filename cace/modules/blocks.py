from typing import Callable, Union, Optional

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

__all__ = ["Dense", "ResidualBlock", "AtomicEnergiesBlock"]

class Dense(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = nn.Identity(),
        weight_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
    ):
        """
        Fully connected linear layer with an optional activation function.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If False, the layer will not have a bias term.
            activation (Callable or nn.Module): Activation function. Defaults to Identity.
            weight_init (Callable): Function to initialize weights.
            bias_init (Callable): Function to initialize bias.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        super().__init__(in_features, out_features, bias)

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

class ResidualBlock(nn.Module):
    """
    A residual block with two dense layers and a skip connection.
    
    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        activation: Activation function to be used in the dense layers.
    """
    def __init__(self, in_features, out_features, activation):
        super().__init__()
        # First dense layer
        self.dense1 = Dense(in_features, out_features, activation=activation)
        # Second dense layer
        self.dense2 = Dense(out_features, out_features, activation=activation)
        # Skip connection with optional dimension matching
        self.skip = nn.Sequential(
            Dense(in_features, out_features, activation=None),
            nn.BatchNorm1d(out_features)
        ) if in_features != out_features else nn.Identity()

    def forward(self, x):
        # Apply skip connection
        identity = self.skip(x)
        # Forward through dense layers
        out = self.dense1(x)
        out = self.dense2(out)
        # Add skip connection result
        out += identity
        return out

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
