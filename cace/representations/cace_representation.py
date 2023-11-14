import torch
from torch import nn
from typing import Callable, Dict, Sequence
from ..tools import elementwise_multiply_3tensors
from ..tools import scatter_sum
from ..modules import NodeEncoder, NodeEmbedding, AngularComponent, AngularComponent_GPU
from ..modules import get_edge_node_type 
from ..modules import find_combo_vectors_nu2, find_combo_vectors_nu3, find_combo_vectors_nu4
from ..modules import Symmetrizer #symmetrize_A_basis

__all__ = ["Cace"]

class Cace(nn.Module):

    def __init__(
        self,
        zs: Sequence[int],
        n_atom_basis: int,
        edge_coding: nn.Module,
        cutoff: float,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        max_l: int,
        max_nu: int,
        device: torch.device = torch.device("cpu")
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            cutoff: cutoff radius
            max_l: the maximum l considered in the angular basis
            max_nu: the maximum correlation order
        """
        super().__init__()
        self.zs = zs
        self.nz = len(zs) # number of elements
        self.n_atom_basis = n_atom_basis
        self.cutoff = cutoff
        self.max_l = max_l
        self.max_nu = max_nu

        # layers
        self.node_onehot = NodeEncoder(self.zs)
        self.node_embedding = NodeEmbedding(
                         node_dim=self.nz, embedding_dim=self.n_atom_basis
                         )
        self.edge_coding = edge_coding
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.device = device
        if self.device  == torch.device("cpu"):
            self.angular_basis = AngularComponent(self.max_l)
        else:
            self.angular_basis = AngularComponent_GPU(self.max_l)

        if max_nu > 4:
            raise NotImplementedError("max_nu > 4 is not supported yet.")         
        self.vec_dict_allnu = {}
        self.vec_dict_allnu[2], _, _  = find_combo_vectors_nu2(self.max_l)
        self.vec_dict_allnu[3], _, _  = find_combo_vectors_nu3(self.max_l)
        self.vec_dict_allnu[4], _, _  = find_combo_vectors_nu4(self.max_l)

        self.l_list = None
        self.symmetrizer = None

    def forward(
        self, 
        data: Dict[str, torch.Tensor]
    ):
        # check if all elements included in self.zs
        
        # setup
        #data["positions"].requires_grad_(True)
        n_nodes = data['positions'].shape[0]
        try:
            num_graphs = data["ptr"].numel() - 1
        except:
            num_graphs = 1

        # Embeddings
        ## code each node/element in one-hot way
        node_one_hot = self.node_onehot(data['atomic_numbers'])

        ## embed to a different dimension
        node_embedded = self.node_embedding(node_one_hot)

        ## get the edge type
        edge_type = get_edge_node_type(edge_index=data["edge_index"], 
                              node_type=node_embedded)
        encoded_edges = self.edge_coding(edge_type)
        
        # compute angular and radial terms
        radial_component = self.radial_basis(data["edge_lengths"]) * self.cutoff_fn(data["edge_lengths"])
        angular_component = self.angular_basis(data["edge_vectors"])

        if self.l_list == None:
            self.l_list = self.angular_basis.get_lxlylz_list()

        # combine
        edge_attri = elementwise_multiply_3tensors(
                      radial_component,
                      angular_component,
                      encoded_edges
        )

        # sum over edge features to each node
        data["node_feat_A"] = scatter_sum(src=edge_attri, 
                                  index=data.edge_index[1], 
                                  dim=0, 
                                  dim_size=n_nodes)

        # symmetrized B basis
        if self.symmetrizer == None:
            self.symmetrizer = Symmetrizer(self.max_nu, self.vec_dict_allnu, self.l_list)
        data["node_feat_B"] = self.symmetrizer.symmetrize_A_basis(node_attr=data["node_feat_A"])

        return data
