from typing import Dict, Union, Sequence, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Dense, ResidualBlock
from ..tools import scatter_sum

__all__ = ["Atomwise", "build_mlp"]

class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(
        self,
        n_in: Optional[int] = None,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        residual: bool = False,
        aggregation_mode: str = "sum",
        output_key: str = "CACE_energy",
        per_atom_output_key: Optional[str] = None,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            aggregation_mode: one of {sum, avg} (default: sum)
            output_key: the key under which the result will be stored
            per_atom_output_key: If not None, the key under which the per-atom result will be stored
        """
        super().__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key

        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        self.n_out = n_out

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.aggregation_mode = aggregation_mode
        self.residual = residual

        if n_in is not None:
            self.outnet = build_mlp(
                n_in=self.n_in,
                n_out=self.n_out,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                residual=residual,
                )
            self.outnet = self.outnet.to(features.device)
        else:
            self.outnet = None

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # reshape the feature vectors
        features = data['node_feats']
        features = features.reshape(features.shape[0], -1)

        if self.n_in is None:
            self.n_in = features.shape[1]
        else:
            assert self.n_in == features.shape[1]

        if self.outnet == None:
            self.outnet = build_mlp(
                n_in=self.n_in,
                n_out=self.n_out,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                residual=self.residual,
                )
            self.outnet = self.outnet.to(features.device)

        # predict atomwise contributions
        y = self.outnet(features)

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            data[self.per_atom_output_key] = y

        # aggregate
        if self.aggregation_mode is not None:
            y = scatter_sum(
                src=y, 
                index=data["batch"], 
                dim=0)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / torch.bincount(data['batch'])

        data[self.output_key] = y
        return data


def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = F.silu,
    residual: bool = False,
    last_bias: bool = True,
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
        layers = []
        # Create residual blocks
        for i in range(0, n_layers - 1, 2):
            in_features = n_neurons[i]
            out_features = n_neurons[min(i + 2, len(n_neurons) - 1)]
            layers.append(ResidualBlock(in_features, out_features, activation))
    else:
        # assign a Dense layer (with activation function) to each hidden layer
        layers = [
            Dense(n_neurons[i], n_neurons[i + 1], activation=activation)
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
                bias=last_bias,
            )
        )
    else:
        layers.append(
            Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=last_bias)
        )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net
