import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List

class SharedRadialLinearTransform(nn.Module):
    def __init__(self, 
        radial_dim: int,
        lxlylz_index: Optional[torch.Tensor] = None, 
        channel_dim: Optional[int] = 1,
        radial_embedding_dim: Optional[int] = None, 
        ):
        super().__init__()
        self.radial_dim = radial_dim
        self.radial_embedding_dim = radial_embedding_dim or radial_dim
        self.channel_dim = channel_dim
        if lxlylz_index is not None:
            self.angular_dim_groups = len(lxlylz_index)
            nlxlylz = lxlylz_index[-1, 1]
            l_matrix = torch.zeros(self.angular_dim_groups, nlxlylz)
            for i,index_now in enumerate(lxlylz_index):
                l_matrix[i, index_now[0]:index_now[1]] = 1
        else:
            self.angular_dim_groups = 1
            l_matrix = torch.ones(1, 1)
        self.register_buffer('l_matrix', l_matrix)

        self.weights = nn.Parameter(
            torch.rand([self.radial_dim, self.radial_embedding_dim, self.angular_dim_groups, self.channel_dim])
            ) 

    def forward(self, 
        x: torch.Tensor # shape: [n_nodes, radial_dim]
        ) -> torch.Tensor:

        l_weights = torch.einsum('lm,ablc->abmc', self.l_matrix, self.weights)
        transformed_x = torch.einsum('ia,abmc->ibmc', x, l_weights)
       
        return transformed_x # shape: [n_nodes, radial_embedding_dim, n_lxlylz, channel_dim]
