import torch
import torch.nn as nn
from ..tools import scatter_sum

__all__ = ["SharedInteraction", "Interaction"]

class SharedInteraction(nn.Module):
    """Interaction layer for the message passing network. Shared L channels."""
    def __init__(self, 
                 cutoff: float,
                 max_l: int,
                 radial_embedding_dim: int, 
                 channel_dim: int,
                 mp_norm_factor: float = 1.0, 
                 memory_coef_init: torch.Tensor=torch.tensor(0.25),
                 ): 
        super().__init__()
        self.max_l = max_l
        self.register_buffer('angular_dim_groups', torch.tensor(self._init_angular_dim_groups(max_l), dtype=torch.int64))

        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "mp_norm_factor", torch.tensor(mp_norm_factor, dtype=torch.get_default_dtype())
        )

        self.radial_embedding_dim = radial_embedding_dim
        self.channel_dim = channel_dim

        # generate trainable tensor of size (radial_embedding_dim, channel_dim)
        self.prefactor = nn.ParameterList([
            nn.Parameter(torch.rand(radial_embedding_dim, channel_dim))
            for _ in self.angular_dim_groups
            ])

        # 1/r0, intialize r0 distributed uniformly in (0.5 cutoff, 1.5 cutoff)
        self.invr0 = nn.ParameterList([
            nn.Parameter((1.0 / self.cutoff) * (torch.rand(radial_embedding_dim, channel_dim) + 0.5))
            for _ in self.angular_dim_groups
            ])

        # intialize as 0.25 in the matrix form
        self.memory_coef = nn.ParameterList([
            nn.Parameter(torch.ones(radial_embedding_dim, channel_dim) * memory_coef_init)
            for _ in self.angular_dim_groups
	    ])
        
    def forward(self, 
                node_feat: torch.Tensor, 
                edge_lengths: torch.Tensor,
                radial_cutoff_fn: torch.Tensor, 
                edge_index: torch.Tensor,
               ) -> torch.Tensor:

        # features of the sender nodes
        # Shape: [n_nodes, radial_dim, angular_dim, embedding_dim]
        sender_features = node_feat[edge_index[0]]

        n_nodes, radial_dim, angular_dim, embedding_dim = node_feat.shape
        assert radial_dim == self.radial_embedding_dim
        assert embedding_dim == self.channel_dim

        n_edges = edge_index.shape[1]
        # features of the sender nodes
        # Shape: [n_edges, radial_dim, angular_dim, embedding_dim]
        sender_features = node_feat[edge_index[0]]

        new_node_feat = torch.zeros(n_nodes, radial_dim, angular_dim, embedding_dim,
                             device=node_feat.device, dtype=node_feat.dtype)

        radial_decay = torch.zeros(
            (n_edges, self.radial_embedding_dim, self.channel_dim),
            dtype=torch.get_default_dtype(), device=node_feat.device)

        for index, (prefactor, invr0, memory_coef) in enumerate(zip(self.prefactor, self.invr0, self.memory_coef)):
            i_start = self.angular_dim_groups[index, 0]
            i_end = self.angular_dim_groups[index, 1]
            # Gather all angular dimensions for the current group
            group = torch.arange(i_start, i_end)

            # the influence of the sender nodes decay with distance
            # prefactor * torch.exp(-r / r0) * cutoff_fn
            radial_decay = torch.exp(-1.0 * torch.einsum('ijk,jk->ijk', edge_lengths.view(n_edges, 1, 1), invr0))
            radial_decay = torch.einsum('ijk,jk->ijk', radial_decay, prefactor)
            radial_decay = torch.einsum('ijk,ijk->ijk', radial_decay, radial_cutoff_fn.view(n_edges, 1, 1))
            group_message = torch.einsum('ijlk,ijk->ijlk', sender_features[:, :, group, :], radial_decay) # Shape: [n_nodes, radial_dim, len(group), embedding_dim]
            memory = torch.einsum('ijlk,jk->ijlk', node_feat[:, :, group, :], memory_coef)
            new_node_feat[:, :, group, :] = memory + scatter_sum(src=group_message, index=edge_index[1], dim=0, dim_size=n_nodes) * self.mp_norm_factor

        return new_node_feat

    def _compute_length_lxlylz(self, l):
        return int((l+1)*(l+2)/2)

    def _init_angular_dim_groups(self, max_l):
        angular_dim_groups: List[int] = []
        l_now = 0
        for l in range(max_l+1):
            l_list_atl = [l_now, l_now + self._compute_length_lxlylz(l)]
            angular_dim_groups.append(l_list_atl)
            l_now += self._compute_length_lxlylz(l)
        return angular_dim_groups


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
        # Shape: [n_nodes/edges, radial_dim, angular_dim, embedding_dim]
        sender_features = node_feat[edge_index[0]]

        # the influence of the sender nodes decat with distance
        # prefactor * torch.exp(-r / r0) * cutoff_fn
        radial_decay = torch.exp(-1.0 * torch.einsum('ijk,jk->ijk', edge_lengths.view(n_edges, 1, 1), self.invr0))
        radial_decay = torch.einsum('ijk,jk->ijk', radial_decay, self.prefactor)
        radial_decay = torch.einsum('ijk,ijk->ijk', radial_decay, radial_cutoff_fn.view(n_edges, 1, 1))
        message = torch.einsum('ijlk,ijk->ijlk', sender_features, radial_decay)

        return node_feat * self.memory_coef + scatter_sum(src=message, index=edge_index[1], dim=0, dim_size=n_nodes) * self.mp_norm_factor
