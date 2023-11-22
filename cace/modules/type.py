import torch
import torch.nn as nn
from typing import Sequence

__all__ = [
    'NodeEncoder',
    'NodeEmbedding',
    'EdgeEncoder',
]

from .utils import get_edge_node_type

class NodeEncoder(nn.Module):
    def __init__(self, zs: Sequence[int]):
        super().__init__()
        self.num_classes = len(zs)
        self.register_buffer("index_map", torch.tensor([zs.index(z) if z in zs else -1 for z in range(max(zs) + 1)], dtype=torch.long))

    def forward(self, atomic_numbers) -> torch.Tensor:
        device = atomic_numbers.device

        # Directly convert atomic numbers to indices using the precomputed map
        indices = self.index_map[atomic_numbers]

        # Handle out-of-range atomic numbers by setting indices to zero
        indices[indices < 0] = 0

        # Generate one-hot encoding
        one_hot_encoding = self.to_one_hot(indices.unsqueeze(-1), num_classes=self.num_classes)
        return one_hot_encoding

    def to_one_hot(self, indices: torch.Tensor, num_classes: int) -> torch.Tensor:
        shape = indices.shape[:-1] + (num_classes,)
        oh = torch.zeros(shape, device=indices.device)

        # scatter_ is the in-place version of scatter
        oh.scatter_(dim=-1, index=indices, value=1)
        return oh

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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.mm(data, self.embedding_weights)

class EdgeEncoder(nn.Module):
    def __init__(self, directed=False):
        super().__init__()
        self.directed = directed

    def forward(self,     
               edge_index: torch.Tensor,  # [2, n_edges]
               node_type: torch.Tensor,  # [n_nodes, n_dims]
               ) -> torch.Tensor:
        # Split the edge tensor into two parts for node1 and node2
        node1, node2 = get_edge_node_type(edge_index, node_type)

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
