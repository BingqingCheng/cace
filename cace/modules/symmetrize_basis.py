import torch
import numpy as np

__all__ = ['Symmetrizer']

"""
This class is used to symmetrize the basis functions in the A basis.
The basis functions are symmetrized by taking the product of the basis functions
"""

class Symmetrizer:
    def __init__(self, nu_max: int, vec_dict_allnu: dict, l_list: list):
        if nu_max >= 5:
            raise NotImplementedError

        self.nu_max = nu_max
        self.vec_dict_allnu = vec_dict_allnu

        # Convert elements of l_list to tuples for dictionary keys
        l_list_tuples = [tuple(l) for l in l_list]
        # Create a dictionary to map tuple to index
        self.l_list_indices = {l_tuple: i for i, l_tuple in enumerate(l_list_tuples)}

    def symmetrize_A_basis(self, node_attr: torch.Tensor):
        num_nodes, n_radial, _, n_chanel = node_attr.size()
        n_angular_sym = 1 + np.sum([len(self.vec_dict_allnu[nu]) for nu in range(2, self.nu_max + 1)])
        sym_node_attr = torch.zeros((num_nodes, n_radial, n_angular_sym, n_chanel),
                                    dtype=node_attr.dtype, device=node_attr.device)

        # Directly assign for nu == 1
        sym_node_attr[:, :, 0, :] = node_attr[:, :, 0, :]
        n_sym_node_attr = 1

        for nu in range(2, self.nu_max + 1):
            vec_dict = self.vec_dict_allnu[nu]
            for i, (_, lxlylz_list) in enumerate(vec_dict.items()):
                for item in lxlylz_list:
                    prefactor = item[-1]
                    indices = [self.l_list_indices[tuple(lxlylz)] for lxlylz in item[:-1]]
                    product = torch.prod(node_attr[:, :, indices, :], dim=2)
                    sym_node_attr[:, :, i + n_sym_node_attr, :] += prefactor * product
            n_sym_node_attr += len(vec_dict)

        return sym_node_attr

# Example usage
# symmetrizer = Symmetrizer(nu_max, vec_dict_allnu, l_list)
# result = symmetrizer.symmetrize_A_basis(node_attr)

