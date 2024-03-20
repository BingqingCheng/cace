import torch
import torch.nn as nn
import numpy as np
from .angular_tools import find_combo_vectors_l1l2

class AngularTensorProduct(nn.Module):
    def __init__(self, max_l: int, l_list: list):
        super().__init__()
        self.max_l = max_l

        # Convert elements of l_list to tuples for dictionary keys
        l_list_tuples = [tuple(l) for l in l_list]
        # Create a dictionary to map tuple to index
        l_list_indices = {l_tuple: i for i, l_tuple in enumerate(l_list_tuples)}
        vec_dict = find_combo_vectors_l1l2(self.max_l)

        self._get_indices(vec_dict, l_list_indices)

    def _get_indices(self, vec_dict, l_list_indices):
        self.indice_list = []
        for i, (l3, l1l2_list) in enumerate(vec_dict.items()):
            l3_index = l_list_indices[tuple(l3)]
            for item in l1l2_list:
                prefactor = int(item[-1])
                l1l2indices = [l_list_indices[tuple(lxlylz)] for lxlylz in item[:-1]]
                self.indice_list.append([l3_index, l1l2indices[0], l1l2indices[1], prefactor])
        self.indice_tensor = torch.tensor(self.indice_list) 

    def forward(self, edge_attr1: torch.Tensor, edge_attr2: torch.Tensor):

        num_edges, n_radial, n_angular, n_chanel = edge_attr1.size()
        edge_attr_new = torch.zeros((num_edges, n_radial, n_angular, n_chanel),
                                    dtype=edge_attr1.dtype, device=edge_attr1.device)

        for item in self.indice_tensor:
            l3_index, l1_index, l2_index, prefactor = item[0], item[1], item[2], item[3] 
            edge_attr_new[:, :, l3_index, :] += prefactor * edge_attr1[:, :, l1_index, :] * edge_attr2[:, :, l2_index, :]

        return edge_attr_new
