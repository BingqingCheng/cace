from typing import Optional, Union, Sequence, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Dense, ResidualBlock, build_mlp

__all__ = ['MessageAr', 'MessageArMLP', 'MessageBchi', 'NodeMemory', 'MessageBA']

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
            #radial_decay = torch.exp(-1.0 * torch.einsum('ijk,jk->ijk', edge_lengths.view(n_edges, 1, 1), invr0))
            #radial_decay = torch.einsum('ijk,jk->ijk', radial_decay, prefactor)
            #radial_decay = torch.einsum('ijk,ijk->ijk', radial_decay, radial_cutoff_fn.view(n_edges, 1, 1))
            #message[:, :, group, :] = torch.einsum('ijlk,ijk->ijlk', sender_features[:, :, group, :], radial_decay)
            radial_decay = torch.exp(-1.0 * edge_lengths.view(n_edges, 1, 1) * invr0[None, :, :]) *  prefactor[None, :, :] * radial_cutoff_fn.view(n_edges, 1, 1)
            message[:, :, group, :] = sender_features[:, :, group, :] * radial_decay[:, :, None, :]

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
    """
    def __init__(self,
        n_in: Optional[int] = None,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 1,
        shared_channels: bool = True,
        shared_l: bool = True,
        n_out: Optional[int] = None,
        lxlylz_index: Optional[torch.Tensor] = None,
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
        self.shared_channels = shared_channels
        self.shared_l = shared_l

        if shared_channels:
            self.nc = 1

        if shared_l is False and lxlylz_index is None:
            raise ValueError("lxlylz_index must be provided if shared_l is False")

        if shared_l:
            self.nl = 1
        else:
            self.nl = len(lxlylz_index)
            self.nlxlylz = lxlylz_index[-1, 1]
            l_matrix = torch.zeros(self.nl, self.nlxlylz)
            for i,index_now in enumerate(lxlylz_index):
                l_matrix[i, index_now[0]:index_now[1]] = 1 
            self.register_buffer('l_matrix', l_matrix)

        if shared_channels and shared_l:
            n_out = 1

        if n_in is not None and n_out is not None:
            self.hnet = build_mlp(
                n_in=self.n_in,
                n_out=n_out,
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

        # check if self.shared_l exists, if not, set it to True
        if not hasattr(self, 'shared_l'):
            self.shared_l = True
            self.nl = 1
            self.nc = 1

        if self.n_in is None:
            self.n_in = features.shape[1]
        else:
            assert self.n_in == features.shape[1]

        if self.hnet == None:
            if self.shared_channels:
                n_out = 1
            else:
                self.nc = channel_dim
                n_out = channel_dim
            if self.shared_l is False:
                n_out *= self.nl 

            self.hnet = build_mlp(
                n_in=self.n_in,
                n_out=n_out,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                residual=self.residual,
                use_batchnorm=self.use_batchnorm,
                )
            self.hnet = self.hnet.to(features.device)

        node_weight = self.hnet(features).reshape(n_nodes, self.nl, self.nc) 
        # make the weight for each lxlylz group the same by multiplying the l_matrix
        if self.shared_l is False:
            node_weight = torch.einsum('lm,ilk->imk', self.l_matrix, node_weight) 
        edge_weight = node_weight[edge_index[0]] # shape: [n_edges, 1 or channel_dim]
        #message = torch.einsum('ijlk,ijlk->ijlk', edge_attri, edge_weight[:, None, :, :])
        message = edge_attri * edge_weight[:, None, :,  :]
        return message # shape: [n_edges, radial_dim, angular_dim, channel_dim]


class MessageBA(nn.Module):
    """ another message passing mechanism
    :math
    m_{j \rightarrow  i} = MLP(B_i) A_i
    """
    def __init__(self,
        n_in: Optional[int] = None,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 1,
        shared_channels: bool = True,
        shared_l: bool = True,
        n_out: Optional[int] = None,
        lxlylz_index: Optional[torch.Tensor] = None,
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
        self.shared_channels = shared_channels
        self.shared_l = shared_l

        if shared_channels:
            self.nc = 1

        if shared_l is False and lxlylz_index is None:
            raise ValueError("lxlylz_index must be provided if shared_l is False")

        if shared_l:
            self.nl = 1
        else:
            self.nl = lxlylz_index[-1, 1]
            l_matrix = torch.zeros(self.nl, self.nl)
            for index_now in lxlylz_index:
                l_size = index_now[1] - index_now[0]
                l_matrix[index_now[0]:index_now[1], index_now[0]:index_now[1]] = torch.ones(l_size, l_size) / l_size
            self.register_buffer('l_matrix', l_matrix)

        if shared_channels and shared_l:
            n_out = 1

        if n_in is not None and n_out is not None:
            self.hnet = build_mlp(
                n_in=self.n_in,
                n_out=n_out,
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
                node_feat_A: torch.Tensor, # shape: [n_nodes, radial_dim, l_dim, channel_dim]
                edge_index: torch.Tensor, # shape: [2, n_edges]
                ) -> torch.Tensor:

        n_nodes, radial_dim, angular_dim, channel_dim = node_feat.shape
        features = node_feat.reshape(n_nodes, -1)
        n_edges = edge_index.shape[1]

        if self.n_in is None:
            self.n_in = features.shape[1]
        else:
            assert self.n_in == features.shape[1]

        if self.hnet is None:
            if self.shared_channels:
                n_out = 1
            else:
                self.nc = channel_dim
                n_out = channel_dim
            if self.shared_l is False:
                n_out *= self.nl

            self.hnet = build_mlp(
                n_in=self.n_in,
                n_out=n_out,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                residual=self.residual,
                use_batchnorm=self.use_batchnorm,
                )
            self.hnet = self.hnet.to(features.device)

        node_weight = self.hnet(features).reshape(n_nodes, self.nl, self.nc)
        # make the weight for each lxlylz group the same by multiplying the l_matrix
        if self.shared_l is False:
            node_weight = torch.einsum('lm,ilk->imk', self.l_matrix, node_weight)
        node_feat_A_new = node_feat_A * node_weight[:, None, :, :]
        message = node_feat_A_new[edge_index[0]] 
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
            #node_memory[:, :, group, :] = torch.einsum('ijlk,jk->ijlk', node_feat[:, :, group, :], memory_coef)
            node_memory[:, :, group, :] = node_feat[:, :, group, :] * memory_coef[None, :, None, :]

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

