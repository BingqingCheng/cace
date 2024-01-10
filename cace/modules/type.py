import torch
import torch.nn as nn
from typing import Sequence

__all__ = [
    'NodeEncoder',
    'NodeEmbedding',
    'EdgeEncoder',
    'NodeEncoder_with_interpolation'
]

from .utils import get_edge_node_type

class NodeEncoder(nn.Module):
    def __init__(self, zs: Sequence[int]):
        super().__init__()
        self.num_classes = len(zs)
        self.register_buffer("index_map", torch.tensor([zs.index(z) if z in zs else -1 for z in range(max(zs) + 1)], dtype=torch.int64))

    def forward(self, atomic_numbers) -> torch.Tensor:
        device = atomic_numbers.device

        # Directly convert atomic numbers to indices using the precomputed map
        indices = self.index_map[atomic_numbers]

        # raise an error if there are out-of-range atomic numbers
        if (indices < 0).any():
            raise ValueError(f"Atomic numbers out of range: {atomic_numbers[indices < 0]}")

        # Generate one-hot encoding
        one_hot_encoding = self.to_one_hot(indices.unsqueeze(-1), num_classes=self.num_classes, device=device)

        return one_hot_encoding

    def to_one_hot(self, indices: torch.Tensor, num_classes: int, device=torch.device) -> torch.Tensor:
        shape = indices.shape[:-1] + (num_classes,)
        oh = torch.zeros(shape, device=device)

        # scatter_ is the in-place version of scatter
        oh.scatter_(dim=-1, index=indices, value=1)
        return oh

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes})"
        )

class NodeEncoder_with_interpolation(nn.Module):
    """
    cumstom NodeEncoder.
    if the atomic number is within zs, using one-hot encoding, otherwise use interpolation between two nearest zs.
    """
    def __init__(self, zs: Sequence[int]):
        super().__init__()
        self.num_classes = len(zs)
        self.zs = zs

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        device = atomic_numbers.device
        # map atomic numbers to indices
        encoded = torch.zeros(atomic_numbers.shape[0], self.num_classes, device=device, dtype=torch.float32)
        # interpolate between two nearest zs
        for i, z in enumerate(atomic_numbers):
            if z in self.zs:
                encoded[i, self.zs.index(z)] = 1
            else:
                for j in range(len(self.zs)):
                    if z < self.zs[j]:
                        encoded[i, j-1] = (self.zs[j] - z) / (self.zs[j] - self.zs[j-1])
                        encoded[i, j] = (z - self.zs[j-1]) / (self.zs[j] - self.zs[j-1])
                        #print(z, i, j , encoded[i, j-1], encoded[i, j])
                        break
        return encoded

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
            self.register_buffer("embedding_weights", embedding_weights, dtype=torch.get_default_dtype())

    def reset_parameters(self, embedding_weights):
        nn.init.xavier_uniform_(embedding_weights)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.mm(data, self.embedding_weights)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_classes={self.embedding_weights.shape[0]}, embedding_dim={self.embedding_weights.shape[1]})"
        )

class EdgeEncoder(nn.Module):
    def __init__(self, directed=True):
        super().__init__()
        self.directed = directed

    def forward(self,     
               edge_index: torch.Tensor,  # [2, n_edges]
               node_type: torch.Tensor,  # [n_nodes, n_dims]
               node_type_2: torch.Tensor=None,  # [n_nodes, n_dims]
               ) -> torch.Tensor:
        # Split the edge tensor into two parts for node1 and node2
        node1, node2 = get_edge_node_type(edge_index, node_type, node_type_2)

        if self.directed:
            # Use batched torch.outer for directed edges
            #encoded_edges = torch.bmm(node1.unsqueeze(2), node2.unsqueeze(1)).flatten(start_dim=1)
            encoded_edges = torch.einsum('ki,kj->kij', node1, node2).flatten(start_dim=1)
        else:
            # Sort node1 and node2 along each edge for undirected edges
            min_node, max_node = torch.min(node1, node2), torch.max(node1, node2)
            #encoded_edges = torch.bmm(min_node.unsqueeze(2), max_node.unsqueeze(1)).flatten(start_dim=1)
            encoded_edges = torch.einsum('ki,kj->kij', min_node, max_node).flatten(start_dim=1)

        return encoded_edges

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(directed={self.directed})"
        )
