import torch
import torch.nn as nn
from typing import Sequence

from ..tools import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
)

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
    def __init__(self, node_dim, embedding_dim, random_seed=42):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(node_dim, embedding_dim))
        if random_seed is not None:
	    torch.manual_seed(random_seed)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)

    def forward(self, data):
        return torch.mm(data, self.weights)

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
