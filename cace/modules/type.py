import torch
import torch.nn as nn
from typing import Sequence

from ..tools import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
)

__all__ = [
    'NodeEncoder',
    'NodeEmbedding',
    'EdgeEncoder',
]

class NodeEncoder(nn.Module):
    def __init__(self, zs: Sequence[int]):
        super().__init__()
        self.z_table = AtomicNumberTable(zs)

    def forward(self, atomic_numbers):
        # this uses one-hot encoding for the node attributes
        indices = atomic_numbers_to_indices(atomic_numbers, z_table=self.z_table)
        one_hot_encoding = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(self.z_table),
            )
        return one_hot_encoding

class NodeEmbedding(nn.Module):
    def __init__(self, node_dim:int, embedding_dim:int, trainable=True, random_seed=42):
        super().__init__()
        embedding_weights = torch.Tensor(node_dim, embedding_dim)
        if random_seed is not None:
            torch.manual_seed(random_seed)
        self.reset_parameters(embedding_weights)

        if trainable:
            self.embedding_weights = nn.Parameter(embedding_weights)
        else:
            self.register_buffer("embedding_weights", embedding_weights)

    def reset_parameters(self, embedding_weights):
        nn.init.xavier_uniform_(embedding_weights)

    def forward(self, data):
        return torch.mm(data, self.embedding_weights)

class EdgeEncoder(nn.Module):
    def __init__(self, directed=False):
        super().__init__()
        self.directed = directed

    def forward(self, edge_tensor):
        encoded_edges = []
        for edge in edge_tensor:
            node1, node2 = edge[:,0], edge[:,1]
            if self.directed:
                encoded_edge = torch.outer(node1, node2).flatten()
            else:
                node1, node2 = sorted([node1, node2], key=lambda x: str(x.tolist()))
                # TODO: this can be reduced in dimension
                encoded_edge = torch.outer(node1, node2).flatten()
            encoded_edges.append(encoded_edge)
        return torch.stack(encoded_edges, dim=0)
