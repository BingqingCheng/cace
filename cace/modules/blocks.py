from typing import Callable, Union, Optional

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

__all__ = ["Dense", "ResidualBlock", "AtomicEnergiesBlock"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable

class Dense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = nn.Identity(),
        use_batchnorm: bool = False,
    ):
        """
        Fully connected linear layer with an optional activation function and batch normalization.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If False, the layer will not have a bias term.
            activation (Callable or nn.Module): Activation function. Defaults to Identity.
            use_batchnorm (bool): If True, include a batch normalization layer.
        """
        super().__init__()
        self.use_batchnorm = use_batchnorm

        # Dense layer
        self.linear = nn.Linear(in_features, out_features, bias)

        # Activation function
        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

        # Batch normalization layer
        if self.use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_features)

    def forward(self, input: torch.Tensor):
        y = self.linear(input)
        if self.use_batchnorm:
            y = self.batchnorm(y)
        y = self.activation(y)
        return y


class ResidualBlock(nn.Module):
    """
    A residual block with flexible number of dense layers, optional batch normalization, 
    and a skip connection.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        activation: Activation function to be used in the dense layers.
        skip_interval: Number of layers between each skip connection.
        use_batchnorm: Boolean indicating whether to use batch normalization.
    """
    def __init__(self, in_features, out_features, activation, skip_interval=2, use_batchnorm=True):
        super().__init__()
        self.skip_interval = skip_interval
        self.use_batchnorm = use_batchnorm
        self.layers = nn.ModuleList()

        # Create dense layers with optional batch normalization
        for _ in range(skip_interval):
            self.layers.append(Dense(in_features, out_features, activation=activation))
            if self.use_batchnorm:
                self.layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features  # Update in_features for the next layer

        # Skip connection with optional dimension matching and batch normalization
        if in_features != out_features:
            skip_layers = [Dense(in_features, out_features, activation=None)]
            if self.use_batchnorm:
                skip_layers.append(nn.BatchNorm1d(out_features))
            self.skip = nn.Sequential(*skip_layers)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = x

        # Forward through dense layers with skip connections and optional batch normalization
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if (i + 1) % self.skip_interval == 0:
                out += identity
                identity = self.skip(out)

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
