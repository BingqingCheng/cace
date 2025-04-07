###############################################
# This module contains functions to compute the angular part of the 
# edge basis functions
###############################################

import numpy as np
import torch
import torch.nn as nn
from math import factorial
from collections import OrderedDict
from typing import Dict, List

__all__=['AngularComponent', 'AngularComponent_GPU', 'make_lxlylz_list', 'make_lxlylz', 'make_l_dict', 'l_dict_to_lxlylz_list', 'compute_length_lxlylz', 'compute_length_lmax', 'compute_length_lmax_numerical', 'lxlylz_factorial_coef', 'lxlylz_factorial_coef_torch', 'l1l2_factorial_coef']

import torch
import torch.nn as nn
from collections import OrderedDict

class AngularComponent(nn.Module):
    """ Angular component of the edge basis functions
        Optimized for CPU usage (use recursive formula)
    """
    def __init__(self, l_max):
        super().__init__()
        self.l_max = l_max
        self.precompute_lxlylz()

    def precompute_lxlylz(self):
        self.lxlylz_dict: Dict[int, List[str]] = OrderedDict({l: [] for l in range(self.l_max + 1)}) #revised
        self.lxlylz_dict[0] = ["0_0_0"] #revised
        for l in range(1, self.l_max + 1):
            for prev_lxlylz_combination in self.lxlylz_dict[l - 1]:
                lxlylz = list(map(int, prev_lxlylz_combination.split('_'))) #revised, string to list
                for i in range(3):
                    lxlylz_combination = lxlylz.copy()
                    lxlylz_combination[i] += 1
                    lxlylz_combination_str = "_".join(map(str, lxlylz_combination))  # revised, list to string
                    if lxlylz_combination_str not in self.lxlylz_dict[l]:
                        self.lxlylz_dict[l].append(lxlylz_combination_str) #revised
        self.lxlylz_list: List(str) = self._convert_lxlylz_to_list() #revised
        # get the start and the end index of the lxlylz_list for each l
        self.lxlylz_index = torch.zeros((self.l_max+1, 2), dtype=torch.long)
        for l in range(self.l_max+1):
            self.lxlylz_index[l, 0] = 0 if l == 0 else self.lxlylz_index[l-1, 1] 
            self.lxlylz_index[l, 1] = compute_length_lmax(l) 

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:

        computed_values: Dict[str, torch.Tensor] = {"0_0_0": torch.ones(vectors.size(0), device=vectors.device, dtype=vectors.dtype)} #revised
        for l in range(1, self.l_max + 1):
            for lxlylz_combination in self.lxlylz_dict[l]:
                parts: List[int] = []
                current_num = ''
                for c in lxlylz_combination:
                    if c == '_':
                        parts.append(int(current_num))
                        current_num = ''
                    else:
                        current_num += c
                parts.append(int(current_num))  # add the last number

                max_val = max(parts)
                i = parts.index(max_val)
                parts[i] -= 1
                prev_key = f"{parts[0]}_{parts[1]}_{parts[2]}" # key in string format
                current_key = lxlylz_combination
                computed_values[current_key] = computed_values[prev_key] * vectors[:, i]
        computed_values_list = self._convert_computed_values_to_list(computed_values)
        return torch.stack(computed_values_list, dim=1)

    def _convert_lxlylz_to_list(self):
        lxlylz_list = []
        for l, combinations in self.lxlylz_dict.items():
            lxlylz_list.extend(combinations)
        return lxlylz_list

    def _convert_computed_values_to_list(self, 
                                         computed_values: Dict[str, torch.Tensor]
                                         ) -> List[torch.Tensor]:
        return [computed_values[comb] for comb in self.lxlylz_list]

    def get_lxlylz_list(self):
        if self.lxlylz_list is None:
            raise ValueError("You must call forward before getting lxlylz_list")
        return self.lxlylz_list
     ##added##
    def get_lxlylz_list_str(self):
        if self.lxlylz_list is None:
            raise ValueError("You must call forward before getting lxlylz_list")
        string_list = ['_'.join(map(str, tpl)) for tpl in self.lxlylz_list ]
        return string_list

    def get_lxlylz_dict(self):
        return self.lxlylz_dict
 
    def get_lxlylz_index(self):
        return self.lxlylz_index

    def __repr__(self):
        return f"AngularComponent(l_max={self.l_max})"

class AngularComponent_GPU(nn.Module):
    """ Angular component of the edge basis functions 
        This version runs faster on gpus but slower on cpus
        The ordering of lxlylz_list is different from the CPU version
    """
    def __init__(self, l_max):
        super().__init__()
        self.l_max = l_max
        self.lxlylz_dict = make_l_dict(l_max)
        self.lxlylz_list = l_dict_to_lxlylz_list(self.lxlylz_dict)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:

        lxlylz_tensor = torch.tensor(self.lxlylz_list, device=vectors.device, dtype=vectors.dtype)

        # Expand vectors and lxlylz_tensor for broadcasting
        vectors_expanded = vectors[:, None, :]  # Shape: [N, 1, 3]
        lxlylz_expanded = lxlylz_tensor[None, :, :]  # Shape: [1, M, 3]

        # Compute terms using broadcasting
        # Each vector component is raised to the power of corresponding lx, ly, lz
        # Somehow this is causing trouble on gpus when doing second order derivatives!!!
        terms = vectors_expanded ** lxlylz_expanded  # Shape: [N, M, 3]

        # Multiply across the last dimension (x^lx * y^ly * z^lz) for each term
        computed_terms = torch.prod(terms, dim=-1)  # Shape: [N, M]
        # to avoid the mps problem with cumprod
        #computed_terms = terms[:, :, 0] * terms[:, :, 1] * terms[:, :, 2]

        return computed_terms

    def get_lxlylz_list(self):
        return self.lxlylz_list

    def get_lxlylz_dict(self):
        return self.lxlylz_dict

    def __repr__(self):
        return f"AngularComponent_GPU(l_max={self.l_max})"

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
    #return torch.tensor(lxlylz, dtype=torch.int64)
    return lxlylz

def make_lxlylz_str(l):
    """
    make a list of lxlylz such that lx + ly + lz = l
    """
    lxlylz = []
    for lx in range(l+1):
        for ly in range(l+1):
            lz = l - lx - ly
            if lz >= 0:
                lxlylz.append('_'.join(map(str, tuple([lx, ly, lz]))))
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
    return lxlylz_list

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


def l1l2_factorial_coef(l1, l2):
    # Ensure inputs are integers
    if not all(isinstance(n, int) for n in l1):
        raise ValueError("All elements of l1 must be integers.")
    if not all(isinstance(n, int) for n in l2):
        raise ValueError("All elements of l2 must be integers.")

    # Compute the multinomial coefficient
    result = 1
    for l1i, l2i in zip(l1, l2):
        result *= factorial(l1i + l2i)
        result /= factorial(l1i)
        result /= factorial(l2i)
    return result

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

def lxlylz_factorial_coef_torch(lxlylz) -> torch.Tensor:

    # Check if lxlylz is a tensor, if not, convert to tensor
    if not isinstance(lxlylz, torch.Tensor):
        lxlylz = torch.tensor(lxlylz, dtype=torch.int64)

    if not torch.all(lxlylz == lxlylz.int()):
        raise ValueError("All elements of lxlylz must be integers.")

    sorted_lxlylz, _ = torch.sort(lxlylz, descending=True)
    l = torch.sum(sorted_lxlylz)

    result = torch.tensor(1, dtype=torch.int)

    for i in torch.arange(int(sorted_lxlylz[0])):
        result = result * (l - i)
        result = (result / (i + 1)).floor()

    for i in torch.arange(1, len(sorted_lxlylz)):
        for j in torch.arange(int(sorted_lxlylz[i])):
            result = result * (l - sorted_lxlylz[:i].sum() - j)
            result = (result / (j + 1)).floor()

    return result
