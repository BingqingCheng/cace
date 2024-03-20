import itertools
import torch
from .angular import l1l2_factorial_coef, lxlylz_factorial_coef, make_lxlylz

__all__ = ['find_combo_vectors_l1l2', 'find_combo_vectors_nu1', 'find_combo_vectors_nu2', 'find_combo_vectors_nu3', 'find_combo_vectors_nu4', 'find_combo_vectors_nu5']

"""
We can store the values

vec_dict_allnu = {}
vec_dict_allnu[2], _, _  = find_combo_vectors_nu2(l_max)
vec_dict_allnu[3], _, _  = find_combo_vectors_nu3(l_max)
vec_dict_allnu[4], _, _  = find_combo_vectors_nu4(l_max)

import pickle

# Save dictionary using pickle
with open('symmetrize_angular_l_list.pickle', 'wb') as handle:
    pickle.dump(vec_dict_allnu, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Read dictionary back
with open('symmetrize_angular_l_list.pickle', 'rb') as handle:
    read_dict = pickle.load(handle)
"""

def no_repeats(*lists):
    combined = sum(lists, [])
    return len(combined) == len(set(combined))

def find_combo_vectors_l1l2(l_max):
    vec_dict = {}
    for lx1, ly1, lz1 in itertools.product(range(l_max+1), repeat=3):
        l1 = lx1 + ly1 + lz1
        if (lx1 + ly1 + lz1) <= l_max:
            for lx2, ly2, lz2 in itertools.product(range(l_max+1), repeat=3):
                l2 = lx2 + ly2 + lz2
                if (lx2 + ly2 + lz2) <= l_max:
                    lx3, ly3, lz3 = lx1 + lx2, ly1 + ly2, lz1 + lz2
                    if (lx3 + ly3 + lz3) <= l_max:
                        prefactor = l1l2_factorial_coef([lx1, ly1, lz1],[lx2, ly2, lz2])
                        key = (lx3, ly3, lz3)
                        vec_dict[key] = vec_dict.get(key, []) + [([lx1, ly1, lz1], [lx2, ly2, lz2], prefactor)]
    return vec_dict

def find_combo_vectors_nu1():
    vector_groups = [0, 0, 0]
    prefactors = 1
    vec_dict = {0: [([0, 0, 0], 1)]}
    return vec_dict, vector_groups, prefactors

def find_combo_vectors_nu2(l_max):
    vector_groups = []
    prefactors = []
    vec_dict = {}
    
    L_list = range(1, l_max+1)
    for i, L in enumerate(L_list):
        for lxlylz_now in make_lxlylz(L):
            lx, ly, lz = lxlylz_now
            prefactor = lxlylz_factorial_coef(lxlylz_now)
            key = L
            vec_dict[key] = vec_dict.get(key, []) + [(lxlylz_now, lxlylz_now, prefactor)]

    vectors = []
    prefactors = []
    vector_idx = []
    # Convert vec_dict to a tensor-friendly format
    vec_dict_tensors = {}
    for i, (key, vec_lists) in enumerate(vec_dict.items()):
        for value in vec_lists:
            [lxlylz_now, lxlylz_now, prefactor] = value
            vector_idx.append(i)
            lxlylz_tensor1 = torch.tensor(lxlylz_now, dtype=torch.int64)
            lxlylz_tensor2 = torch.tensor(lxlylz_now, dtype=torch.int64)
            vectors.append(torch.stack([lxlylz_tensor1, lxlylz_tensor2]))
            prefactors.append(prefactor)

    if not vectors:
        # Return an empty tensor
        stacked_vectors = torch.tensor([])
    else:
        stacked_vectors = torch.stack(vectors)

    return vec_dict, stacked_vectors, torch.tensor(vector_idx, dtype=torch.int64), torch.tensor(prefactors, dtype=torch.int64), len(vec_dict)

def find_combo_vectors_nu3(l_max):
    vec_dict = {}
    for lx1, ly1, lz1 in itertools.product(range(l_max+1), repeat=3):
        l1 = lx1 + ly1 + lz1
        if 0 < (lx1 + ly1 + lz1) <= l_max:
            for lx2, ly2, lz2 in itertools.product(range(l_max+1), repeat=3):
                l2 = lx2 + ly2 + lz2
                if (lx1 + ly1 + lz1) <= (lx2 + ly2 + lz2) <= l_max:
                    lx3, ly3, lz3 = lx1 + lx2, ly1 + ly2, lz1 + lz2
                    if (lx3 + ly3 + lz3) <= l_max:
                        prefactor = lxlylz_factorial_coef([lx1, ly1, lz1])*lxlylz_factorial_coef([lx2, ly2, lz2])
                        key = (l1 ,l2)
                        vec_dict[key] = vec_dict.get(key, []) + [([lx1, ly1, lz1], [lx2, ly2, lz2], [lx3, ly3, lz3], prefactor)]

    vectors = []
    prefactors = []
    vector_idx = []
    # Convert vec_dict to a tensor-friendly format
    vec_dict_tensors = {}
    for i, (key, vec_lists) in enumerate(vec_dict.items()):
        for value in vec_lists:
            [lxlylz_now1, lxlylz_now2, lxlylz_now3, prefactor] = value
            vector_idx.append(i)
            lxlylz_tensor1 = torch.tensor(lxlylz_now1, dtype=torch.int64)
            lxlylz_tensor2 = torch.tensor(lxlylz_now2, dtype=torch.int64)
            lxlylz_tensor3 = torch.tensor(lxlylz_now3, dtype=torch.int64)
            vectors.append(torch.stack([lxlylz_tensor1, lxlylz_tensor2, lxlylz_tensor3]))
            prefactors.append(prefactor)
           
    if not vectors:
        # Return an empty tensor
        stacked_vectors = torch.tensor([])
    else:
        stacked_vectors = torch.stack(vectors)
 
    return vec_dict, stacked_vectors, torch.tensor(vector_idx, dtype=torch.int64), torch.tensor(prefactors, dtype=torch.int64), len(vec_dict)

def find_combo_vectors_nu4(l_max):
    vec_dict = {}
    for lx1, ly1, lz1 in itertools.product(range(l_max + 1), repeat=3):
        l1 = lx1 + ly1 + lz1
        if 0 < l1 <= l_max:
            for lx2, ly2, lz2 in itertools.product(range(l_max + 1), repeat=3):
                l2 = lx2 + ly2 + lz2
                if l1 < l2 <= l_max:  # Ensuring l2 is strictly greater than l1
                    for dx, dy, dz in itertools.product(range(l_max + 1), repeat=3):
                        dl = dx + dy + dz
                        if dl >= 1:
                            lx3, ly3, lz3 = lx1 + dx, ly1 + dy, lz1 + dz
                            lx4, ly4, lz4 = lx2 + dx, ly2 + dy, lz2 + dz
                            if (lx3 + ly3 + lz3) <= l_max and (lx4 + ly4 + lz4) <= l_max:
                                prefactor = lxlylz_factorial_coef([lx1, ly1, lz1]) \
                                    *lxlylz_factorial_coef([lx2, ly2, lz2]) \
                                    *lxlylz_factorial_coef([dx, dy, dz])
                                key = (l1 ,l2, dl)
                                vec_dict[key] = vec_dict.get(key, []) + \
                                [([lx1, ly1, lz1], [lx2, ly2, lz2], [lx3, ly3, lz3], [lx4, ly4, lz4], prefactor)]

    vectors = []
    prefactors = []
    vector_idx = []
    # Convert vec_dict to a tensor-friendly format
    vec_dict_tensors = {}
    for i, (key, vec_lists) in enumerate(vec_dict.items()):
        for value in vec_lists:
            [lxlylz_now1, lxlylz_now2, lxlylz_now3, lxlylz_now4, prefactor] = value
            vector_idx.append(i)
            lxlylz_tensor1 = torch.tensor(lxlylz_now1, dtype=torch.int64)
            lxlylz_tensor2 = torch.tensor(lxlylz_now2, dtype=torch.int64)
            lxlylz_tensor3 = torch.tensor(lxlylz_now3, dtype=torch.int64)
            lxlylz_tensor4 = torch.tensor(lxlylz_now4, dtype=torch.int64)
            vectors.append(torch.stack([lxlylz_tensor1, lxlylz_tensor2, lxlylz_tensor3, lxlylz_tensor4]))
            prefactors.append(prefactor)
           
    if not vectors:
        # Return an empty tensor
        stacked_vectors = torch.tensor([])
    else:
        stacked_vectors = torch.stack(vectors)
 
    return vec_dict, stacked_vectors, torch.tensor(vector_idx, dtype=torch.int64), torch.tensor(prefactors, dtype=torch.int64), len(vec_dict)


def find_combo_vectors_nu5(l_max):
    vec_dict = {}
    for lx1, ly1, lz1 in itertools.product(range(l_max + 1), repeat=3):
        l1 = lx1 + ly1 + lz1
        if 0 < l1 <= l_max:
            for lx2, ly2, lz2 in itertools.product(range(l_max + 1), repeat=3):
                l2 = lx2 + ly2 + lz2
                if l1 < l2 <= l_max:  # Ensuring l2 is strictly greater than l1
                    for dx, dy, dz in itertools.product(range(l_max + 1), repeat=3):
                        dl = dx + dy + dz
                        if dl >= 1:
                            lx3, ly3, lz3 = lx1 + dx, ly1 + dy, lz1 + dz
                            l3 = lx3 + ly3 + lz3
                            if l3 <= l_max:
                                for dx2, dy2, dz2 in itertools.product(range(l_max + 1), repeat=3):
                                    dl2 = dx2 + dy2 + dz2
                                    if dl2 >= 1:
                                        lx4, ly4, lz4 = lx2 + dx, ly2 + dy, lz2 + dz
                                        l4 = lx4 + ly4 + lz4
                                        lx5, ly5, lz5 = dx2 + dx, dy2 + dy, dz2 + dz
                                        l5 = lx5 + ly5 + lz5
                                        if l4 <= l_max and l5 <= l_max:
                                            prefactor = lxlylz_factorial_coef([lx1, ly1, lz1]) \
                                                *lxlylz_factorial_coef([lx2, ly2, lz2]) \
                                                *lxlylz_factorial_coef([dx, dy, dz]) \
                                                *lxlylz_factorial_coef([dx2, dy2, dz2])
                                            # check there are no repeats in (l1, l2, l3, l4, l5) 
                                            if no_repeats([l1, l2, l3, l4, l5]):  
                                                key = (l1 ,l2, dl, dl2)
                                                vec_dict[key] = vec_dict.get(key, []) + \
                                                    [([lx1, ly1, lz1], [lx2, ly2, lz2], [lx3, ly3, lz3], [lx4, ly4, lz4], [lx5, ly5, lz5], prefactor)]

    vectors = []
    prefactors = []
    vector_idx = []
    # Convert vec_dict to a tensor-friendly format
    vec_dict_tensors = {}
    for i, (key, vec_lists) in enumerate(vec_dict.items()):
        for value in vec_lists:
            [lxlylz_now1, lxlylz_now2, lxlylz_now3, lxlylz_now4, lxlylz_now5, prefactor] = value
            vector_idx.append(i)
            lxlylz_tensor1 = torch.tensor(lxlylz_now1, dtype=torch.int64)
            lxlylz_tensor2 = torch.tensor(lxlylz_now2, dtype=torch.int64)
            lxlylz_tensor3 = torch.tensor(lxlylz_now3, dtype=torch.int64)
            lxlylz_tensor4 = torch.tensor(lxlylz_now4, dtype=torch.int64)
            lxlylz_tensor5 = torch.tensor(lxlylz_now5, dtype=torch.int64)
            vectors.append(torch.stack([lxlylz_tensor1, lxlylz_tensor2, lxlylz_tensor3, lxlylz_tensor4, lxlylz_tensor5]))
            prefactors.append(prefactor)

    if not vectors:
        # Return an empty tensor
        stacked_vectors = torch.tensor([])
    else:
        stacked_vectors = torch.stack(vectors)

    return vec_dict, stacked_vectors, torch.tensor(vector_idx, dtype=torch.int64), torch.tensor(prefactors, dtype=torch.int64), len(vec_dict)

def cal_num_B_features(l_m=8):
    """function to precalculate the number of B features"""
    n_B_feat_dict = {}

    for l_max in range(1,l_m):

        vec_dict_allnu = {}

        n_B_feat = 0
        vec_dict_allnu[1], _, _  = find_combo_vectors_nu1()
        n_B_feat += len(vec_dict_allnu[1])
        n_B_feat_dict[(l_max, 1)] = n_B_feat

        vec_dict_allnu[2], _, _  = find_combo_vectors_nu2(l_max)
        n_B_feat += len(vec_dict_allnu[2])
        n_B_feat_dict[(l_max, 2)] = n_B_feat

        vec_dict_allnu[3], _, _  = find_combo_vectors_nu3(l_max)
        n_B_feat += len(vec_dict_allnu[3])
        n_B_feat_dict[(l_max, 3)] = n_B_feat

        vec_dict_allnu[4], _, _  = find_combo_vectors_nu4(l_max)
        n_B_feat += len(vec_dict_allnu[4])
        n_B_feat_dict[(l_max, 4)] = n_B_feat

        vec_dict_allnu[5], _, _  = find_combo_vectors_nu5(l_max)
        n_B_feat += len(vec_dict_allnu[5])
        n_B_feat_dict[(l_max, 5)] = n_B_feat
    return n_B_feat_dict
