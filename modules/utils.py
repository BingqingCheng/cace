###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import List, Optional, Tuple

import torch

def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths

def get_edge_node_type(
    edge_index: torch.Tensor,  # [2, n_edges]
    node_type: torch.Tensor,  # [n_nodes, n_dims]
) -> torch.Tensor:
    edge_type = torch.zeros([edge_index.shape[1], node_type.shape[1], 2])
    sender_type = node_type[edge_index[0]]
    receiver_type = node_type[edge_index[1]]
    edge_type[:, :, 0] = sender_type
    edge_type[:, :, 1] = receiver_type
    return edge_type  # [n_edges, n_dims, 2]

