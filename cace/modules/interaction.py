import torch
import torch.nn as nn
from ..tools import scatter_sum

__all__ = ["Interaction"]

class Interaction(nn.Module):
    """Interaction layer for the message passing network."""
    def __init__(self, 
                 cutoff: float,
                 radial_embedding_dim: int, 
                 channel_dim: int,
                 mp_norm_factor: float = 1.0, 
                 memory_coef: torch.Tensor=torch.tensor(0.25), 
                 trainable: bool = True):
        super().__init__()
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "mp_norm_factor", torch.tensor(mp_norm_factor, dtype=torch.get_default_dtype())
        )

        self.radial_embedding_dim = radial_embedding_dim
        self.channel_dim = channel_dim

        if not isinstance(memory_coef, torch.Tensor):
            memory_coef = torch.tensor(memory_coef, dtype=torch.get_default_dtype())

        if trainable:
            self.memory_coef = nn.Parameter(memory_coef)
        else:
            self.register_buffer("memory_coef", memory_coef, dtype=torch.get_default_dtype())

        # TODO: maybe add seperate L channels
        # generate trainable tensor of size (radial_embedding_dim, channel_dim)
        self.prefactor = nn.Parameter(
            torch.rand(radial_embedding_dim, channel_dim))
        # 1/r0, intialize r0 distributed uniformly in (0.5 cutoff, 1.5 cutoff)
        self.invr0 = nn.Parameter(
            (1.0 / self.cutoff) * (torch.rand(radial_embedding_dim, channel_dim) + 0.5))
        
    def forward(self, 
                node_feat: torch.Tensor, 
                edge_lengths: torch.Tensor,
                radial_cutoff_fn: torch.Tensor, 
                edge_index: torch.Tensor,
               ) -> torch.Tensor:

        n_edges = edge_index.shape[1]
        n_nodes = node_feat.shape[0]

        radial_decay = torch.zeros(
            (n_edges, self.radial_embedding_dim, self.channel_dim), 
            dtype=torch.get_default_dtype(), device=node_feat.device)

        # features of the sender nodes
        # Shape: [n_nodes, radial_dim, angular_dim, embedding_dim]
        sender_features = node_feat[edge_index[0]]

        # the influence of the sender nodes decat with distance
        # prefactor * torch.exp(-r / r0) * cutoff_fn
        radial_decay = torch.exp(-1.0 * torch.einsum('ijk,jk->ijk', edge_lengths.view(n_edges, 1, 1), self.invr0))
        radial_decay = torch.einsum('ijk,jk->ijk', radial_decay, self.prefactor)
        radial_decay = torch.einsum('ijk,ijk->ijk', radial_decay, radial_cutoff_fn.view(n_edges, 1, 1))
        message = torch.einsum('ijlk,ijk->ijlk', sender_features, radial_decay)

        return node_feat * self.memory_coef + scatter_sum(src=message, index=edge_index[1], dim=0, dim_size=n_nodes) * self.mp_norm_factor
