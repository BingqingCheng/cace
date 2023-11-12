###############################################
# This module contains functions to compute the angular part of the 
# edge basis functions
###############################################

import numpy as np
import torch
import torch.nn as nn
from math import factorial
from collections import OrderedDict

class AngularComponent(nn.Module):
    def __init__(self, l_max):
        super().__init__()
        self.l_max = l_max
        self.lxlylz_dict = OrderedDict({l: [] for l in range(l_max + 1)})
        self.lxlylz_list = None

    def forward(self, vectors):
        if not isinstance(vectors, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if vectors.dim() != 2 or vectors.size(1) != 3:
            raise ValueError("Input tensor must have shape [N, 3]")
        
        self.lxlylz_dict = OrderedDict({l: [] for l in range(self.l_max + 1)})
        self.lxlylz_list = None
        N, _ = vectors.shape
        self.lxlylz_dict[0] = [[0, 0, 0]]
        computed_values = {(0, 0, 0): torch.ones(N, device=vectors.device)}

        self._recursive_compute(self.l_max, vectors, computed_values)
        self.lxlylz_list = self._convert_lxlylz_to_list()
        computed_values_list = self._convert_computed_values_to_list(computed_values)
        return computed_values_list

    def _recursive_compute(self, l, vectors, computed_values):
        if l == 0:
            return
        if not self.lxlylz_dict[l]:
            self._recursive_compute(l - 1, vectors, computed_values)
            for prev_lxlylz_combination in self.lxlylz_dict[l - 1]:
                for i in range(3):
                    lxlylz_combination = prev_lxlylz_combination.copy()
                    lxlylz_combination[i] += 1
                    lxlylz_combination_tuple = tuple(lxlylz_combination)
                    if lxlylz_combination_tuple not in computed_values:
                        self.lxlylz_dict[l].append(lxlylz_combination)
                        computed_values[lxlylz_combination_tuple] = computed_values[tuple(prev_lxlylz_combination)] * vectors[:, i]

    def _convert_lxlylz_to_list(self):
        lxlylz_list = []
        for l, combinations in self.lxlylz_dict.items():
            for lxlylz_combination in combinations:
                lxlylz_list.append(lxlylz_combination)
        return lxlylz_list

    def _convert_computed_values_to_list(self, computed_values):
        computed_values_list = []
        for l, combinations in self.lxlylz_dict.items():
            for lxlylz_combination in combinations:
                computed_values_list.append(computed_values[tuple(lxlylz_combination)])
        return torch.stack(computed_values_list, dim=1)

    def get_lxlylz_list(self):
        if self.lxlylz_list is None:
            raise ValueError("You must call forward before getting lxlylz_list")
        return self.lxlylz_list

    def get_lxlylz_dict(self):
        return self.lxlylz_dict


class AngularComponent_old(nn.Module):
    """ Angular component of the edge basis functions """
    def __init__(self, l_list: torch.Tensor):
        super().__init__()
        self.l_list = l_list

    def forward(self, vector_list: torch.Tensor):
        computed_list = []
        
        for vector in vector_list:
            x, y, z = vector
            terms_list = []
            
            for l in self.l_list:
                lx, ly, lz = l
                term = (x ** lx) * (y ** ly) * (z ** lz)
                terms_list.append(term)
            
            computed_list.append(torch.stack(terms_list))
        
        return torch.stack(computed_list)

    def get_l_list(self):
        return self.l_list

def make_lxlylz_list(l_max: int):
    """
    make a list of lxlylz up to l_max
    """
    l_dict = make_l_dict(l_max)
    return l_dict_to_lxlylz_list(l_dict) 

def l_index_select(l):
    """ select the index of the lxlylz_list based on l """
    return np.arange(compute_length_lmax(l-1), compute_length_lmax(l))

def make_lxlylz(l):
    """
    make a list of lxlylz such that lx + ly + lz = l
    """
    lxlylz = []
    for lx in range(l+1):
        for ly in range(l+1):
            lz = l - lx - ly
            if lz >= 0:
                lxlylz.append([lx, ly, lz])
    return lxlylz

def make_l_dict(l_max):
    """
    make a ordered dictionary of lxlylz list
    up to l_max
    """
    l_dict = OrderedDict()
    for l in range(l_max+1):
        l_dict[l] = make_lxlylz(l)
    return l_dict

def l_dict_to_lxlylz_list(l_dict):
    """
    convert the ordered dictionary to a list of lxlylz
    """
    lxlylz_list = []
    for l in l_dict:
        lxlylz_list += l_dict[l]
    return torch.tensor(lxlylz_list)

def compute_length_lxlylz(l):
    """ compute the length of the lxlylz list based on l """
    return int((l+1)*(l+2)/2)

def compute_length_lmax(l_max):
    """ compute the length of the lxlylz list based on l_max """
    return int((l_max+1)*(l_max+2)*(l_max+3)/6)

def compute_length_lmax_numerical(l_max):
    """ compute the length of the lxlylz list based on l_max numerically"""
    length = 0
    for l in range(l_max+1):
        length += compute_length_lxlylz(l)
    return length


def lxlylz_factorial_coef(lxlylz):
    # Ensure inputs are integers
    if not all(isinstance(n, int) for n in lxlylz):
        raise ValueError("All elements of lxlylz must be integers.")

    # Sort the elements in descending order
    sorted_lxlylz = sorted(lxlylz, reverse=True)

    # Compute the sum l = lx + ly + lz
    l = sum(sorted_lxlylz)

    # Compute the multinomial coefficient
    result = factorial(l)
    for lxly in sorted_lxlylz:
        result //= factorial(lxly)

    return result

def lxlylz_factorial_coef_torch(lxlylz):
    # Ensure inputs are integers
    if not torch.all(lxlylz == lxlylz.int()):
        raise ValueError("All elements of lxlylz must be integers.")

    # Sort the elements in descending order
    sorted_lxlylz, _ = torch.sort(lxlylz, descending=True)

    # Compute the sum l = lx + ly + lz
    l = torch.sum(sorted_lxlylz)

    # The result starts at 1 and will be built up by multiplication
    result = torch.tensor(1, dtype=torch.int)

    # Compute the multinomial coefficient
    for i in range(int(sorted_lxlylz[0])):
        result *= (l - i)
        result //= (i + 1)

    for i in range(1, len(sorted_lxlylz)):
        for j in range(int(sorted_lxlylz[i])):
            result *= (l - sorted_lxlylz[:i].sum() - j)
            result //= (j + 1)

    return result
