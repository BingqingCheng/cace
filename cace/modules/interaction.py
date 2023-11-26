import torch
import torch.nn as nn
from ..tools import scatter_sum
from .radial import ExponentialDecayRBF

__all__ = ["Interaction"]

class Interaction(nn.Module):
    """Interaction layer for the message passing network."""
    def __init__(self, 
                 cutoff: float,
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
        self.radial_filter = ExponentialDecayRBF(n_rbf=1, cutoff=cutoff, trainable=trainable)

        if not isinstance(memory_coef, torch.Tensor):
            memory_coef = torch.tensor(memory_coef, dtype=torch.get_default_dtype())

        if trainable:
            self.memory_coef = nn.Parameter(memory_coef)
        else:
            self.register_buffer("memory_coef", memory_coef, dtype=torch.get_default_dtype())

    def forward(self, 
                node_feat: torch.Tensor, 
                edge_lengths: torch.Tensor,
                radial_cutoff: torch.Tensor, 
                edge_index: torch.Tensor,
                n_nodes: int
               ) -> torch.Tensor:

        # features of the sender nodes
        sender_features = node_feat[edge_index[0]]
        # the influence of the sender nodes decat with distance
        radial_decay = self.radial_filter(edge_lengths) * radial_cutoff
        message = sender_features * radial_decay.view(sender_features.shape[0], 1, 1, 1)
        return node_feat * self.memory_coef + scatter_sum(src=message, index=edge_index[1], dim=0, dim_size=n_nodes) * self.mp_norm_factor
