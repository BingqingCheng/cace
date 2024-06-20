from typing import Callable, Union, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

__all__ = ["build_mlp", "Dense", "ResidualBlock", "AtomicEnergiesBlock"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable

def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = F.silu,
    residual: bool = False,
    use_batchnorm: bool = False,
    bias: bool = True,
    last_zero_init: bool = False,
) -> nn.Module:
    """
    Build multiple layer fully connected perceptron neural network.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
        residual: whether to use residual connections between layers
        use_batchnorm: whether to use batch normalization between layers
    """
    # get list of number of nodes in input, hidden & output layers
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for i in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        # get list of number of nodes hidden layers
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
        n_neurons = [n_in] + n_hidden + [n_out]

    if residual:
        if n_layers < 3 or n_layers % 2 == 0:
            raise ValueError("Residual networks require at least 3 layers and an odd number of layers")
        layers = []
        # Create residual blocks every 2 layers
        for i in range(0, n_layers - 1, 2):
            in_features = n_neurons[i]
            out_features = n_neurons[min(i + 2, len(n_neurons) - 1)]
            layers.append(
                ResidualBlock(
                    in_features,
                    out_features,
                    activation,
                    skip_interval=2,
                    use_batchnorm=use_batchnorm,
                    )
               )
    else:
        # assign a Dense layer (with activation function) to each hidden layer
        layers = [
            Dense(n_neurons[i], n_neurons[i + 1], activation=activation, use_batchnorm=use_batchnorm, bias=bias)
            for i in range(n_layers - 1)
        ]

    # assign a Dense layer (without activation function) to the output layer

    if last_zero_init:
        layers.append(
            Dense(
                n_neurons[-2],
                n_neurons[-1],
                activation=None,
                weight_init=torch.nn.init.zeros_,
                bias=bias,
            )
        )
    else:
        layers.append(
            Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=bias)
        )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net

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

        # Skip connection with optional dimension matching and batch normalization
        if in_features != out_features:
            skip_layers = [Dense(in_features, out_features, activation=None)]
            if self.use_batchnorm:
                skip_layers.append(nn.BatchNorm1d(out_features))
            self.skip = nn.Sequential(*skip_layers)
        else:
            self.skip = nn.Identity()

        # Create dense layers with optional batch normalization
        for _ in range(skip_interval):
            self.layers.append(Dense(in_features, out_features, activation=activation))
            if self.use_batchnorm:
                self.layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features  # Update in_features for the next layer

    def forward(self, x):
        identity = self.skip(x)
        out = x

        # Forward through dense layers with skip connections and optional batch normalization
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if (i + 1) % self.skip_interval == 0:
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
