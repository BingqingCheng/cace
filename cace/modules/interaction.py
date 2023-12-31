from typing import Optional, Union, Sequence, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Dense, ResidualBlock, build_mlp

__all__ = ['MessageAr', 'MessageArMLP', 'MessageBchi', 'NodeMemory']

class MessageAr(nn.Module):
    """
    Interaction layer for the message passing network. 
    Dependent on radial and channel dimensions, shared L channels.
    :math
    m_{j \rightarrow  i, kn\mathbf{l}}^{(t)} = 
    f(r_{ji})
    A_{j, kn\mathbf{l}}^{(t)}
    """
    def __init__(self, 
                 cutoff: float,
                 max_l: int,
                 radial_embedding_dim: int, 
                 channel_dim: int,
                 ): 
        super().__init__()
        self.register_buffer('angular_dim_groups', torch.tensor(init_angular_dim_groups(max_l), dtype=torch.int64))

        self.radial_embedding_dim = radial_embedding_dim
        self.channel_dim = channel_dim

        # generate trainable tensor of size (radial_embedding_dim, channel_dim)
        self.prefactor = nn.ParameterList([
            nn.Parameter(torch.rand(radial_embedding_dim, channel_dim))
            for _ in self.angular_dim_groups
            ])

        # 1/r0, intialize r0 distributed uniformly in (0.5 cutoff, 1.5 cutoff)
        self.invr0 = nn.ParameterList([
            nn.Parameter((1.0 / cutoff) * (torch.rand(radial_embedding_dim, channel_dim) + 0.5))
            for _ in self.angular_dim_groups
            ])

    def forward(self, 
                node_feat: torch.Tensor, # shape: [n_nodes, radial_dim, angular_dim, channel_dim]
                edge_lengths: torch.Tensor, # shape: [n_edges]
                radial_cutoff_fn: torch.Tensor, # shape: [n_edges]
                edge_index: torch.Tensor, # shape: [2, n_edges]
               ) -> torch.Tensor:

        n_nodes, radial_dim, angular_dim, channel_dim = node_feat.shape
        assert radial_dim == self.radial_embedding_dim
        assert channel_dim == self.channel_dim

        n_edges = edge_index.shape[1]
        # features of the sender nodes
        # Shape: [n_edges, radial_dim, angular_dim, channel_dim]
        sender_features = node_feat[edge_index[0]]

        message = torch.zeros(
            (n_edges, radial_dim, angular_dim, channel_dim),
            device=node_feat.device, dtype=node_feat.dtype)

        radial_decay = torch.zeros(
            (n_edges, radial_dim, channel_dim),
            device=node_feat.device, dtype=node_feat.dtype)

        for index, (prefactor, invr0) in enumerate(zip(self.prefactor, self.invr0)):
            i_start = self.angular_dim_groups[index, 0]
            i_end = self.angular_dim_groups[index, 1]
            # Gather all angular dimensions for the current group
            group = torch.arange(i_start, i_end)

            # the influence of the sender nodes decay with distance
            # prefactor * torch.exp(-r / r0) * cutoff_fn
            radial_decay = torch.exp(-1.0 * torch.einsum('ijk,jk->ijk', edge_lengths.view(n_edges, 1, 1), invr0))
            radial_decay = torch.einsum('ijk,jk->ijk', radial_decay, prefactor)
            radial_decay = torch.einsum('ijk,ijk->ijk', radial_decay, radial_cutoff_fn.view(n_edges, 1, 1))
            message[:, :, group, :] = torch.einsum('ijlk,ijk->ijlk', sender_features[:, :, group, :], radial_decay)

        return message # shape: [n_edges, radial_dim, angular_dim, channel_dim]

class MessageArMLP(nn.Module):
    """
    Interaction layer for the message passing network.
    Dependent on radial and channel dimensions, shared L channels.
    :math
    m_{j \rightarrow  i, kn\mathbf{l}}^{(t)} =
    f(r_{ji})
    A_{j, kn\mathbf{l}}^{(t)}
    """
    def __init__(self,
                 cutoff: float,
                 max_l: int,
                 radial_embedding_dim: int,
                 channel_dim: int,
                 activation: Union[Callable, nn.Module] = F.sigmoid,
                 ):
        super().__init__()
        self.register_buffer('angular_dim_groups', torch.tensor(init_angular_dim_groups(max_l), dtype=torch.int64))

        self.radial_embedding_dim = radial_embedding_dim
        self.channel_dim = channel_dim

        # generate trainable tensor of size (radial_embedding_dim, channel_dim)
        self.weights = nn.ParameterList([
            nn.Parameter(
                torch.rand(radial_embedding_dim, channel_dim)
                )
            for _ in self.angular_dim_groups
            ])

        self.activation = activation

    def forward(self,
                node_feat: torch.Tensor, # shape: [n_nodes, radial_dim, angular_dim, channel_dim]
                radial_component: torch.Tensor, # shape: [n_edges, radial_dim]
                radial_cutoff_fn: torch.Tensor, # shape: [n_edges]
                edge_index: torch.Tensor, # shape: [2, n_edges]
               ) -> torch.Tensor:

        n_nodes, radial_dim, angular_dim, channel_dim = node_feat.shape
        assert channel_dim == self.channel_dim

        n_edges = edge_index.shape[1]
        # features of the sender nodes
        # Shape: [n_edges, radial_dim, angular_dim, channel_dim]
        sender_features = node_feat[edge_index[0]]

        message = torch.zeros(
            (n_edges, radial_dim, angular_dim, channel_dim),
            device=node_feat.device, dtype=node_feat.dtype)

        radial_decay = torch.zeros(
            (n_edges, radial_dim, channel_dim),
            device=node_feat.device, dtype=node_feat.dtype)

        for index, weights in enumerate(self.weights):
            i_start = self.angular_dim_groups[index, 0]
            i_end = self.angular_dim_groups[index, 1]
            # Gather all angular dimensions for the current group
            group = torch.arange(i_start, i_end)

            # the influence of the sender nodes decay with distance
            # radial_component * weights * cutoff_fn
            radial_decay = torch.einsum('ai,ik->ak', radial_component, weights)
            radial_decay = self.activation(radial_decay)
            radial_decay = radial_decay * radial_cutoff_fn.view(n_edges, 1)
            message[:, :, group, :] = sender_features[:, :, group, :] * radial_decay.unsqueeze(1).unsqueeze(2)

        return message # shape: [n_edges, radial_dim, angular_dim, channel_dim]

class MessageBchi(nn.Module):
    """ another message passing mechanism
    :math
    m_{j \rightarrow  i, kn\mathbf{l}}^{(t)} = 
    \sum_{l_x, l_y, l_z}^{l_x+l_y+l_z = l} \dfrac{l!}{l_x ! l_y ! l_z !}
    \chi_{kn\mathbf{l}} (\sigma_i, \sigma_{i'})
    h(B_{j, k\{nl\}}),

    in general the h function can be dependent on the radial and channel dimenstion and shared across l channels
    For now we use the same h function (MLP) for all features
    """
    def __init__(self,
        n_in: Optional[int] = None,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 1,
        activation: Callable = F.silu,
        residual: bool = False,
        use_batchnorm: bool = False,
        ):

        super().__init__()

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.residual = residual
        self.use_batchnorm = use_batchnorm

        if n_in is not None:
            self.hnet = build_mlp(
                n_in=self.n_in,
                n_out=1,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                residual=self.residual,
                use_batchnorm=self.use_batchnorm,
                )
        else:
            self.hnet = None

    def forward(self,
                node_feat: torch.Tensor, # shape: [n_nodes, radial_dim, angular_dim, channel_dim]
                edge_attri: torch.Tensor, # shape: [n_edges, radial_dim, angular_dim, channel_dim]
                edge_index: torch.Tensor, # shape: [2, n_edges]
		) -> torch.Tensor:

        n_nodes, radial_dim, angular_dim, channel_dim = node_feat.shape
        features = node_feat.reshape(n_nodes, -1)
        n_edges = edge_index.shape[1]

        if self.n_in is None:
            self.n_in = features.shape[1]
        else:
            assert self.n_in == features.shape[1]

        if self.hnet == None:
            self.hnet = build_mlp(
                n_in=self.n_in,
                n_out=1,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                residual=self.residual,
                use_batchnorm=self.use_batchnorm,
                )
            self.hnet = self.hnet.to(features.device)

        node_weight = self.hnet(features) # shape: [n_nodes, 1]
        edge_weight = node_weight[edge_index[0]] # shape: [n_edges, 1]
        message = torch.einsum('ijlk,ijlk->ijlk', edge_attri, edge_weight.view(n_edges, 1, 1, 1))
        return message # shape: [n_edges, radial_dim, angular_dim, channel_dim]


class NodeMemory(nn.Module):
    """ Compute the memory of the node during message passing """
    def __init__(self,
                 max_l: int,
                 radial_embedding_dim: int,
                 channel_dim: int,
                 memory_coef_init: torch.Tensor=torch.tensor(0.25),
                 ):
        super().__init__()
        self.max_l = max_l
        self.register_buffer('angular_dim_groups', torch.tensor(init_angular_dim_groups(max_l), dtype=torch.int64))
        self.radial_embedding_dim = radial_embedding_dim
        self.channel_dim = channel_dim

        # generate trainable tensor of size (radial_embedding_dim, channel_dim)
        # intialize as 0.25 in the matrix form
        self.memory_coef = nn.ParameterList([
            nn.Parameter(torch.ones(radial_embedding_dim, channel_dim) * memory_coef_init)
            for _ in self.angular_dim_groups
            ])

    def forward(self, 
                node_feat: torch.Tensor,
               ) -> torch.Tensor:

        n_nodes, radial_dim, angular_dim, channel_dim = node_feat.shape
        assert radial_dim == self.radial_embedding_dim
        assert channel_dim == self.channel_dim

        node_memory = torch.zeros_like(node_feat)

        for index, memory_coef in enumerate(self.memory_coef):
            i_start = self.angular_dim_groups[index, 0]
            i_end = self.angular_dim_groups[index, 1]
            # Gather all angular dimensions for the current group
            group = torch.arange(i_start, i_end)
            node_memory[:, :, group, :] = torch.einsum('ijlk,jk->ijlk', node_feat[:, :, group, :], memory_coef)

        return node_memory # shape: [n_nodes, radial_dim, angular_dim, channel_dim]

def compute_length_lxlylz(l):
    return int((l+1)*(l+2)/2)

def init_angular_dim_groups(max_l):
    angular_dim_groups: List[int] = []
    l_now = 0
    for l in range(max_l+1):
        length_at_l = compute_length_lxlylz(l)
        l_list_atl = [l_now, l_now + length_at_l]
        angular_dim_groups.append(l_list_atl)
        l_now += length_at_l 
    return angular_dim_groups

