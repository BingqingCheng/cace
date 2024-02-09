import torch
from torch import nn
from typing import Callable, Dict, Sequence, Optional, List, Any

from ..tools import torch_geometric
from ..tools import elementwise_multiply_3tensors, scatter_sum
from ..modules import (
    NodeEncoder, 
    NodeEmbedding, 
    EdgeEncoder,
    AngularComponent, 
    AngularComponent_GPU,
    SharedRadialLinearTransform,
    Symmetrizer,
    #Symmetrizer_JIT,
    MessageAr, 
    MessageBchi,
    NodeMemory
    )
from ..modules import (
    get_edge_vectors_and_lengths,
    ) 

__all__ = ["Cace"]

class Cace(nn.Module):

    def __init__(
        self,
        zs: Sequence[int],
        n_atom_basis: int,
        cutoff: float,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        max_l: int,
        max_nu: int,
        num_message_passing: int,
        type_message_passing: List[str] = ["M", "Ar", "Bchi"],
        args_message_passing: Dict[str, Any] = {"M": {}, "Ar": {}, "Bchi": {}},
        embed_receiver_nodes: bool = False,
        atom_embedding_random_seed: List[int] = [42, 42], 
        n_radial_basis: Optional[int] = None,
        avg_num_neighbors: float = 10.0,
        device: torch.device = torch.device("cpu"),
        timeit: bool = False,
    ):
        """
        Args:
            zs: list of atomic numbers
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            edge_coding: layer for encoding edge type
            cutoff: cutoff radius
            radial_basis: layer for expanding interatomic distances in a basis set
            n_radial_basis: number of radial embedding dimensions
            cutoff_fn: cutoff function
            cutoff: cutoff radius
            max_l: the maximum l considered in the angular basis
            max_nu: the maximum correlation order
            num_message_passing: number of message passing layers
            avg_num_neighbors: average number of neighbors per atom, used for normalization
        """
        super().__init__()
        self.zs = zs
        self.nz = len(zs) # number of elements
        self.n_atom_basis = n_atom_basis
        self.cutoff = cutoff
        self.max_l = max_l
        self.max_nu = max_nu
        self.mp_norm_factor = 1.0/(avg_num_neighbors)**0.5 # normalization factor for message passing

        # layers
        self.node_onehot = NodeEncoder(self.zs)
        # sender node embedding
        self.node_embedding_sender = NodeEmbedding(
                         node_dim=self.nz, embedding_dim=self.n_atom_basis, random_seed=atom_embedding_random_seed[0]
                         )
        if embed_receiver_nodes:
            self.node_embedding_receiver = NodeEmbedding(
                         node_dim=self.nz, embedding_dim=self.n_atom_basis, random_seed=atom_embedding_random_seed[1]
                         )
        else:
            self.node_embedding_receiver = self.node_embedding_sender 

        self.edge_coding = EdgeEncoder(directed=True) 
        self.n_edge_channels = n_atom_basis**2

        self.radial_basis = radial_basis
        self.n_radial_func = self.radial_basis.n_rbf
        self.n_radial_basis = n_radial_basis or self.radial_basis.n_rbf
        self.cutoff_fn = cutoff_fn
        # The AngularComponent_GPU version sometimes has trouble with second derivatives
        #if self.device  == torch.device("cpu"):
        #    self.angular_basis = AngularComponent(self.max_l)
        #else:
        #    self.angular_basis = AngularComponent_GPU(self.max_l)
        self.angular_basis = AngularComponent(self.max_l)
        radial_transform = SharedRadialLinearTransform(
                                max_l=self.max_l,
                                radial_dim=self.n_radial_func,
                                radial_embedding_dim=self.n_radial_basis,
                                channel_dim=self.n_edge_channels
                                )
        self.radial_transform = radial_transform
        #self.radial_transform = torch.jit.script(radial_transform)

        self.l_list = self.angular_basis.get_lxlylz_list()
        self.symmetrizer = Symmetrizer(self.max_nu, self.max_l, self.l_list)
        # the JIT version seems to be slower
        #symmetrizer = Symmetrizer_JIT(self.max_nu, self.max_l, self.l_list)
        #self.symmetrizer = torch.jit.script(symmetrizer)

        # for message passing layers
        self.num_message_passing = num_message_passing
        self.message_passing_list = nn.ModuleList([
            nn.ModuleList([
                NodeMemory(
                    max_l=self.max_l,
                    radial_embedding_dim=self.n_radial_basis,
                    channel_dim=self.n_edge_channels,
                    **args_message_passing["M"] if "M" in args_message_passing else {}
                    ) if "M" in type_message_passing else None,

                MessageAr(
                    cutoff=cutoff,
                    max_l=self.max_l,
                    radial_embedding_dim=self.n_radial_basis,
                    channel_dim=self.n_edge_channels,
                    **args_message_passing["Ar"] if "Ar" in args_message_passing else {}
                    ) if "Ar" in type_message_passing else None,

                MessageBchi(
                    lxlylz_index = self.angular_basis.get_lxlylz_index(),
                    **args_message_passing["Bchi"] if "Bchi" in args_message_passing else {}
                    ) if "Bchi" in type_message_passing else None,
            ]) 
            for _ in range(self.num_message_passing)
            ])


        self.device = device

    def forward(
        self, 
        data: Dict[str, torch.Tensor]
    ):
        # setup
        n_nodes = data['positions'].shape[0]
        if data["batch"] == None:
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=self.device)
        else:
            batch_now = data["batch"]

        node_feats_list = []

        # Embeddings
        ## code each node/element in one-hot way
        node_one_hot = self.node_onehot(data['atomic_numbers'])
        ## embed to a different dimension
        node_embedded_sender = self.node_embedding_sender(node_one_hot)
        node_embedded_receiver = self.node_embedding_receiver(node_one_hot)
        ## get the edge type
        encoded_edges = self.edge_coding(edge_index=data["edge_index"],
                                         node_type=node_embedded_sender,
                                         node_type_2=node_embedded_receiver,)

        # compute angular and radial terms
        edge_vectors, edge_lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
            normalize=True,
            )
        radial_component = self.radial_basis(edge_lengths) 
        radial_cutoff = self.cutoff_fn(edge_lengths)
        angular_component = self.angular_basis(edge_vectors)

        # combine
        # 4-dimensional tensor: [n_edges, radial_dim, angular_dim, embedding_dim]
        edge_attri = elementwise_multiply_3tensors(
                      radial_component * radial_cutoff,
                      angular_component,
                      encoded_edges
        )

        # sum over edge features to each node
        # 4-dimensional tensor: [n_nodes, radial_dim, angular_dim, embedding_dim]
        node_feat_A = scatter_sum(src=edge_attri, 
                                  index=data["edge_index"][1], 
                                  dim=0, 
                                  dim_size=n_nodes)

        # mix the different radial components
        node_feat_A = self.radial_transform(node_feat_A)

        # symmetrized B basis
        node_feat_B = self.symmetrizer(node_attr=node_feat_A)
        node_feats_list.append(node_feat_B)

        # message passing
        for nm, mp_Ar, mp_Bchi in self.message_passing_list: 
            if nm is not None:
                momeory_now = nm(node_feat=node_feat_A)
            else:
                momeory_now = 0.0

            if mp_Bchi is not None:
                message_Bchi = mp_Bchi(node_feat=node_feat_B,
                    edge_attri=edge_attri,
                    edge_index=data["edge_index"],
                    )
                node_feat_A_Bchi = scatter_sum(src=message_Bchi,
                                       index=data["edge_index"][1],
                                       dim=0,
                                       dim_size=n_nodes)
                # mix the different radial components
                node_feat_A_Bchi = self.radial_transform(node_feat_A_Bchi)
            else:
                node_feat_A_Bchi = 0.0 

            if mp_Ar is not None:
                message_Ar = mp_Ar(node_feat=node_feat_A,
                    edge_lengths=edge_lengths,
                    radial_cutoff_fn=radial_cutoff,
                    edge_index=data["edge_index"],
                    )

                node_feat_Ar = scatter_sum(src=message_Ar,
                                  index=data["edge_index"][1],
                                  dim=0,
                                  dim_size=n_nodes)
            else:
                node_feat_Ar = 0.0
 
            node_feat_A = node_feat_Ar + node_feat_A_Bchi 
            node_feat_A *= self.mp_norm_factor
            node_feat_A += momeory_now
            node_feat_B = self.symmetrizer(node_attr=node_feat_A)
            node_feats_list.append(node_feat_B)
     
        node_feats_out = torch.stack(node_feats_list, dim=-1)

        try:
            displacement = data["displacement"]
        except:
            displacement = None

        return {
            "positions": data["positions"],
            "cell": data["cell"],
            "displacement": displacement,
            "batch": batch_now,
            "node_feats": node_feats_out,
            #"node_feats_A": node_feat_A
        }
