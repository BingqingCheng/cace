import time
import torch
from torch import nn
from typing import Callable, Dict, Sequence, Optional

from ..tools import torch_geometric
from ..tools import elementwise_multiply_3tensors, scatter_sum
from ..modules import (
    NodeEncoder, 
    NodeEmbedding, 
    AngularComponent, 
    AngularComponent_GPU,
    SharedRadialLinearTransform,
    Symmetrizer,
    Symmetrizer_JIT,
    MessageAr, 
    MesssageBchi,
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
        edge_coding: nn.Module,
        cutoff: float,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        max_l: int,
        max_nu: int,
        num_message_passing: int,
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
        self.node_embedding = NodeEmbedding(
                         node_dim=self.nz, embedding_dim=self.n_atom_basis
                         )
        self.edge_coding = edge_coding
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
        self.message_passing_list = [
            nn.ModuleList([
                NodeMemory(
                    max_l=self.max_l,
                    radial_embedding_dim=self.n_radial_basis,
                    channel_dim=self.n_edge_channels,
                    ),
                MessageAr(
                    cutoff=cutoff,
                    max_l=self.max_l,
                    radial_embedding_dim=self.n_radial_basis,
                    channel_dim=self.n_edge_channels,
                    ),
                MesssageBchi()
            ]) 
            for _ in range(self.num_message_passing)
            ]


        self.device = device
        self.timeit = timeit

    def forward(
        self, 
        data: Dict[str, torch.Tensor]
    ):
        # TODO: check if all elements included in self.zs
        
        # setup
        n_nodes = data['positions'].shape[0]
        if data["batch"] == None:
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=self.device)
        else:
            batch_now = data["batch"]

        node_feats_list = []

        # Embeddings
        ## code each node/element in one-hot way
        # add timing to each step of the forward pass
        t0 = time.time()
        node_one_hot = self.node_onehot(data['atomic_numbers'])
        t1 = time.time()
        if self.timeit: print("node_one_hot time: {}".format(t1-t0))

        ## embed to a different dimension
        node_embedded = self.node_embedding(node_one_hot)
        t2 = time.time()
        if self.timeit: print("node_embedded time: {}".format(t2-t1))

        ## get the edge type
        encoded_edges = self.edge_coding(edge_index=data["edge_index"],
                                         node_type=node_embedded)

        t3 = time.time()        
        if self.timeit: print("encoded_edges time: {}".format(t3-t2))

        # compute angular and radial terms
        edge_vectors, edge_lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
            normalize=True,
            )
        t4 = time.time()
        if self.timeit: print("edge_vectors time: {}".format(t4-t3))

        radial_component = self.radial_basis(edge_lengths) 
        radial_cutoff = self.cutoff_fn(edge_lengths)
        # normalize=False, use the REANN way
        # edge_vectors = edge_vectors * radial_cutoff.view(edge_vectors.shape[0], 1)
        angular_component = self.angular_basis(edge_vectors)
        t5 = time.time()
        if self.timeit: print("radial and angular component time: {}".format(t5-t4))

        # combine
        # 4-dimensional tensor: [n_edges, radial_dim, angular_dim, embedding_dim]
        edge_attri = elementwise_multiply_3tensors(
                      radial_component * radial_cutoff,
                      angular_component,
                      encoded_edges
        )

        t6 = time.time()
        if self.timeit: print("elementwise_multiply_3tensors time: {}".format(t6-t5))

        # sum over edge features to each node
        # 4-dimensional tensor: [n_nodes, radial_dim, angular_dim, embedding_dim]
        node_feat_A = scatter_sum(src=edge_attri, 
                                  index=data["edge_index"][1], 
                                  dim=0, 
                                  dim_size=n_nodes)
        t7 = time.time()
        if self.timeit: print("scatter_sum time: {}".format(t7-t6))

        # mix the different radial components
        node_feat_A = self.radial_transform(node_feat_A)
        t8 = time.time()
        if self.timeit: print("radial_transform time: {}".format(t8-t7))

        # symmetrized B basis
        node_feat_B = self.symmetrizer(node_attr=node_feat_A)
        node_feats_list.append(node_feat_B)

        t9 = time.time()
        if self.timeit: print("symmetrizer time: {}".format(t9-t8))

        # message passing
        for nm, mp_Ar, mp_Bchi in self.message_passing_list: 
            t_mp_start = time.time()
            momeory_now = nm(node_feat=node_feat_A)

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

            message_Ar = mp_Ar(node_feat=node_feat_A,
                edge_lengths=edge_lengths,
                radial_cutoff_fn=radial_cutoff,
                edge_index=data["edge_index"],
                )

            node_feat_A = node_feat_A_Bchi + scatter_sum(src=message_Ar,
                                  index=data["edge_index"][1],
                                  dim=0,
                                  dim_size=n_nodes)
 
            node_feat_A *= self.mp_norm_factor
            node_feat_A += momeory_now
            node_feat_B = self.symmetrizer(node_attr=node_feat_A)
            node_feats_list.append(node_feat_B)
            t_mp_end = time.time()
            if self.timeit: print("message passing time: {}".format(t_mp_end-t_mp_start))
     
        node_feats_out = torch.stack(node_feats_list, dim=-1)

        # displacement is needed for computing stress/virial
        #if isinstance(data, torch_geometric.batch.Batch):
        #    displacement = data.to_dict().get("displacement", None)
        #elif isinstance(data, Dict):
        #    ddisplacement = data.get("displacement", None)
        # or maybe
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
