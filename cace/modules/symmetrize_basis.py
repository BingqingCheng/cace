import torch
import torch.nn as nn
import numpy as np
from .angular_tools import (
    find_combo_vectors_nu2,
    find_combo_vectors_nu3,
    find_combo_vectors_nu4
    )

__all__ = ['Symmetrizer', 'Symmetrizer_JIT', 'Symmetrizer_Tensor']

"""
This class is used to symmetrize the basis functions in the A basis.
The basis functions are symmetrized by taking the product of the basis functions
"""

class Symmetrizer_JIT(nn.Module):
    """ This symmetrizer is implemented in JIT mode. """ 
    def __init__(self, max_nu: int, max_l: int, l_list: torch.Tensor):
        super().__init__()

        if max_nu >= 5:
            raise NotImplementedError

        self.max_nu = max_nu
        self.max_l = max_l

        _, vg2, vi2, pf2, n2 = find_combo_vectors_nu2(max_l)
        self.register_buffer("vector_groups_2", vg2)
        self.register_buffer("vector_idx_2", vi2)
        self.register_buffer("prefactors_2", pf2)
        self.n2_start = 1
        self.n3_start = 1 + n2

        _, vg3, vi3, pf3, n3 = find_combo_vectors_nu3(max_l)
        self.register_buffer("vector_groups_3", vg3)
        self.register_buffer("vector_idx_3", vi3)
        self.register_buffer("prefactors_3", pf3)
        self.n4_start = 1 + n2 + n3

        _, vg4, vi4, pf4, n4 = find_combo_vectors_nu4(max_l)
        self.register_buffer("vector_groups_4", vg4)
        self.register_buffer("vector_idx_4", vi4)
        self.register_buffer("prefactors_4", pf4)

        # Initialize buffers for each nu value
        self.n_angular_sym = 1
        if max_nu >= 2:
            self.n_angular_sym += n2
        if max_nu >= 3:
            self.n_angular_sym += n3
        if max_nu == 4:
            self.n_angular_sym += n4

        # Register l_list as a buffer
        l_list_tensor = torch.tensor([l for l in l_list], dtype=torch.int64)
        self.register_buffer('l_list_tensor', l_list_tensor)

    @torch.jit.export
    def forward(self, node_attr: torch.Tensor) -> torch.Tensor:
        num_nodes, n_radial, _, n_chanel = node_attr.size()
        sym_node_attr = torch.zeros((num_nodes, n_radial, self.n_angular_sym, n_chanel),
                                    dtype=node_attr.dtype, device=node_attr.device)

        # Directly assign for nu == 1
        sym_node_attr[:, :, 0, :] = node_attr[:, :, 0, :]
        n_sym_node_attr = 1
        if self.max_nu >= 2:
            for i, item in enumerate(self.vector_groups_2):
                prefactor = self.prefactors_2[i]
                idx = self.vector_idx_2[i]

                # Convert item to list of tuples
                indices = [self._get_index_from_l_list(lxlylz) for lxlylz in item]
                product = torch.prod(node_attr[:, :, indices, :], dim=2)
                sym_node_attr[:, :, idx + self.n2_start, :] += prefactor * product
        if self.max_nu >= 3:
            for i, item in enumerate(self.vector_groups_3):
                prefactor = self.prefactors_3[i]
                idx = self.vector_idx_3[i]
                indices = [self._get_index_from_l_list(lxlylz) for lxlylz in item]
                product = torch.prod(node_attr[:, :, indices, :], dim=2)
                sym_node_attr[:, :, idx + self.n3_start, :] += prefactor * product
        if self.max_nu == 4:
            for i, item in enumerate(self.vector_groups_4):
                prefactor = self.prefactors_4[i]
                idx = self.vector_idx_4[i]
                indices = [self._get_index_from_l_list(lxlylz) for lxlylz in item]
                product = torch.prod(node_attr[:, :, indices, :], dim=2)
                sym_node_attr[:, :, idx + self.n4_start, :] += prefactor * product
        return sym_node_attr

    @torch.jit.export
    def _get_index_from_l_list(self, lxlylz: torch.Tensor) -> int:
        return torch.where((self.l_list_tensor == lxlylz).all(dim=1))[0][0].item()


class Symmetrizer(nn.Module):
    def __init__(self, max_nu: int, max_l: int, l_list: list):
        super().__init__()
        if max_nu >= 5:
            raise NotImplementedError

        self.max_nu = max_nu
        self.max_l = max_l

        # Convert elements of l_list to tuples for dictionary keys
        l_list_tuples = [tuple(l) for l in l_list]
        # Create a dictionary to map tuple to index
        self.l_list_indices = {l_tuple: i for i, l_tuple in enumerate(l_list_tuples)}

        if max_nu > 4:
            raise NotImplementedError("max_nu > 4 is not supported yet.")
        self.vec_dict_allnu = {}
        if max_nu >= 2:
            self.vec_dict_allnu[2]  = find_combo_vectors_nu2(self.max_l)[0]
        if max_nu >= 3:
            self.vec_dict_allnu[3]  = find_combo_vectors_nu3(self.max_l)[0]
        if max_nu == 4:
            self.vec_dict_allnu[4]  = find_combo_vectors_nu4(self.max_l)[0]

        self.indice_dict_allnu = None 
        self._get_indices_allnu()

    def _get_indices_allnu(self):
        self.indice_dict_allnu = {}
        for nu in range(2, self.max_nu + 1):
            self.indice_dict_allnu[nu] = {}
            for i, (l_key, lxlylz_list) in enumerate(self.vec_dict_allnu[nu].items()):
                self.indice_dict_allnu[nu][l_key] = []
                for item in lxlylz_list:
                    prefactor = item[-1]
                    indices = [self.l_list_indices[tuple(lxlylz)] for lxlylz in item[:-1]]
                    # append to the dictionary
                    self.indice_dict_allnu[nu][l_key].append([indices, prefactor])

    def forward(self, node_attr: torch.Tensor):
        try:
            self.indice_dict_allnu
        except AttributeError:
           self._get_indices_allnu()

        num_nodes, n_radial, _, n_chanel = node_attr.size()
        n_angular_sym = 1 + np.sum([len(self.vec_dict_allnu[nu]) for nu in range(2, self.max_nu + 1)])
        sym_node_attr = torch.zeros((num_nodes, n_radial, n_angular_sym, n_chanel),
                                    dtype=node_attr.dtype, device=node_attr.device)

        # Directly assign for nu == 1
        sym_node_attr[:, :, 0, :] = node_attr[:, :, 0, :]
        n_sym_node_attr = 1

        for nu in range(2, self.max_nu + 1):
            for i, (_, indices_list) in enumerate(self.indice_dict_allnu[nu].items()):
                for item in indices_list:
                    indices, prefactor = item[0], item[-1]
                    product = torch.prod(node_attr[:, :, indices, :], dim=2)
                    # somehow MPS doesn't like torch.prod, as it uses cumprod during autograd.
                    # one can use the following:
                    #product = node_attr[:, :, indices[0], :]
                    #for idx in indices[1:]:
                    #    product = product * node_attr[:, :, idx, :]
                    sym_node_attr[:, :, i + n_sym_node_attr, :] += prefactor * product
            n_sym_node_attr += len(self.indice_dict_allnu[nu])

        return sym_node_attr

class Symmetrizer_Tensor(nn.Module):
    """ This symmetrizer is implemented using tensor operations. 
        Not performant for nu=4, but should be fine for smaller nu and large max_l.
    """
    def __init__(self, max_nu: int, max_l: int, l_list: list):
        super().__init__()
        if max_nu >= 5:
            raise NotImplementedError

        self.max_nu = max_nu
        self.max_l = max_l
        self.l_list = l_list
        self.n_l = len(l_list)

        # Convert elements of l_list to tuples for dictionary keys
        l_list_tuples = [tuple(l) for l in l_list]
        # Create a dictionary to map tuple to index
        self.l_list_indices = {l_tuple: i for i, l_tuple in enumerate(l_list_tuples)}

        if max_nu > 4:
            raise NotImplementedError("max_nu > 4 is not supported yet.")
        self.vec_dict_allnu = {}
        self.sym_tensor_allnu = {}
        if max_nu >= 2:
            self.vec_dict_allnu[2]  = find_combo_vectors_nu2(self.max_l)[0]
            # 3D tensor of shape (n_l, n_l, len(vec_dict_allnu[2]))
            self.sym_tensor_allnu[2] = torch.zeros((self.n_l, self.n_l, len(self.vec_dict_allnu[2])))
            # loop through the dictionary and assign the values to the tensor 
            for i, (_, lxlylz_list) in enumerate(self.vec_dict_allnu[2].items()):
                for item in lxlylz_list:
                    prefactor = item[-1]
                    indices = [self.l_list_indices[tuple(lxlylz)] for lxlylz in item[:-1]]
                    self.sym_tensor_allnu[2][indices[0], indices[1], i] = prefactor 

        if max_nu >= 3:
            self.vec_dict_allnu[3]  = find_combo_vectors_nu3(self.max_l)[0]

            self.sym_tensor_allnu[3] = torch.zeros((self.n_l, self.n_l, self.n_l, len(self.vec_dict_allnu[3])))
            # loop through the dictionary and assign the values to the tensor
            for i, (_, lxlylz_list) in enumerate(self.vec_dict_allnu[3].items()):
                for item in lxlylz_list:
                    prefactor = item[-1]
                    indices = [self.l_list_indices[tuple(lxlylz)] for lxlylz in item[:-1]]
                    self.sym_tensor_allnu[3][indices[0], indices[1], indices[2], i] = prefactor

        if max_nu == 4:
            self.vec_dict_allnu[4]  = find_combo_vectors_nu4(self.max_l)[0]

            self.sym_tensor_allnu[4] = torch.zeros((self.n_l, self.n_l, self.n_l, self.n_l, len(self.vec_dict_allnu[4])))
            # loop through the dictionary and assign the values to the tensor
            for i, (_, lxlylz_list) in enumerate(self.vec_dict_allnu[4].items()):
                for item in lxlylz_list:
                    prefactor = item[-1]
                    indices = [self.l_list_indices[tuple(lxlylz)] for lxlylz in item[:-1]]
                    self.sym_tensor_allnu[4][indices[0], indices[1], indices[2], indices[3], i] = prefactor

    def forward(self, node_attr: torch.Tensor):
        num_nodes, n_radial, n_l, n_chanel = node_attr.size()
        assert n_l == self.n_l

        n_angular_sym = 1 + np.sum([len(self.vec_dict_allnu[nu]) for nu in range(2, self.max_nu + 1)])
        sym_node_attr = torch.zeros((num_nodes, n_radial, n_angular_sym, n_chanel),
                                    dtype=node_attr.dtype, device=node_attr.device)

        # Directly assign for nu == 1
        sym_node_attr[:, :, 0, :] = node_attr[:, :, 0, :]
        n_sym_node_attr = 1

        if self.max_nu >= 2:
            # for nu==2, node_attr_2 is the product of node_attr
            # node_attr: [num_nodes, n_radial, n_l, n_l, n_channel]
            node_attr_2 = torch.einsum('ijlk,ijmk->ijlmk', node_attr, node_attr) 
            sym_tensor = self.sym_tensor_allnu[2]
            n_sym_node_attr_now = sym_tensor.shape[2]

            sym_node_attr[:, :, n_sym_node_attr:n_sym_node_attr+n_sym_node_attr_now: , :] = torch.einsum('ijlmk,lma->ijak', node_attr_2, sym_tensor) 
            n_sym_node_attr += n_sym_node_attr_now

        if self.max_nu >= 3:
            node_attr_3 = torch.einsum('ijlmk,ijnk->ijlmnk', node_attr_2, node_attr)
            sym_tensor = self.sym_tensor_allnu[3]
            n_sym_node_attr_now = sym_tensor.shape[3]

            sym_node_attr[:, :, n_sym_node_attr:n_sym_node_attr+n_sym_node_attr_now: , :] = torch.einsum('ijlmnk,lmna->ijak', node_attr_3, sym_tensor)
            n_sym_node_attr += n_sym_node_attr_now

        if self.max_nu >= 4:
            node_attr_4 = torch.einsum('ijlmnk,ijok->ijlmnok', node_attr_3, node_attr)
            sym_tensor = self.sym_tensor_allnu[4]
            n_sym_node_attr_now = sym_tensor.shape[4]

            sym_node_attr[:, :, n_sym_node_attr:n_sym_node_attr+n_sym_node_attr_now: , :] = torch.einsum('ijlmnok,lmnoa->ijak', node_attr_4, sym_tensor)
            n_sym_node_attr += n_sym_node_attr_now

        return sym_node_attr

