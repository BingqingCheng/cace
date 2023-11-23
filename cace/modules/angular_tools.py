import itertools
import torch
from .angular import lxlylz_factorial_coef, make_lxlylz

__all__ = ['find_combo_vectors_nu1', 'find_combo_vectors_nu2', 'find_combo_vectors_nu3', 'find_combo_vectors_nu4', "n_B_feat_dict"]

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

def find_combo_vectors_nu1():
    vector_groups = [0, 0, 0]
    prefactors = 1
    vec_dict = {0: [([0, 0, 1], 1)]}
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
            #print(prefactor)
            vector_groups.append(lxlylz_now)
            prefactors.append(prefactor)
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

    return vec_dict, torch.stack(vectors), torch.tensor(vector_idx, dtype=torch.int64), torch.tensor(prefactors, dtype=torch.int64), len(vec_dict)

def find_combo_vectors_nu3(l_max):
    vector_groups = []
    prefactors = []
    vec_dict = {}
    
    for lx1, ly1, lz1 in itertools.product(range(l_max+1), repeat=3):
        l1 = lx1 + ly1 + lz1
        if 0 < (lx1 + ly1 + lz1) <= l_max:
            for lx2, ly2, lz2 in itertools.product(range(l_max+1), repeat=3):
                l2 = lx2 + ly2 + lz2
                if (lx1 + ly1 + lz1) <= (lx2 + ly2 + lz2) <= l_max:
                    lx3, ly3, lz3 = lx1 + lx2, ly1 + ly2, lz1 + lz2
                    if (lx3 + ly3 + lz3) <= l_max:
                        if ([lx1, ly1, lz1], [lx2, ly2, lz2], [lx3, ly3, lz3]) not in vector_groups:
                            vector_groups.append(([lx1, ly1, lz1], [lx2, ly2, lz2], [lx3, ly3, lz3]))
                            prefactor = lxlylz_factorial_coef([lx1, ly1, lz1])*lxlylz_factorial_coef([lx2, ly2, lz2])
                            prefactors.append(prefactor)
                            
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
            
    return vec_dict, torch.stack(vectors), torch.tensor(vector_idx, dtype=torch.int64), torch.tensor(prefactors, dtype=torch.int64), len(vec_dict)

def find_combo_vectors_nu4(l_max):
    vector_groups = []
    vec_dict = {}
    prefactors = []
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
                                vector_groups.append(([lx1, ly1, lz1], [lx2, ly2, lz2], 
                                                      [lx3, ly3, lz3], [lx4, ly4, lz4]))
                                prefactor = lxlylz_factorial_coef([lx1, ly1, lz1]) \
                                    *lxlylz_factorial_coef([lx2, ly2, lz2]) \
                                    *lxlylz_factorial_coef([dx, dy, dz])
                                prefactors.append(prefactor)
                                
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
            
    return vec_dict, torch.stack(vectors), torch.tensor(vector_idx, dtype=torch.int64), torch.tensor(prefactors, dtype=torch.int64), len(vec_dict)


# a dictionary storing the number of B features for (l_max, nu_max)
n_B_feat_dict: dict 
n_B_feat_dict = {(1, 1): 1,
     (1, 2): 2,
     (1, 3): 2,
     (1, 4): 2,
     (2, 1): 1,
     (2, 2): 3,
     (2, 3): 4,
     (2, 4): 4,
     (3, 1): 1,
     (3, 2): 4,
     (3, 3): 6,
     (3, 4): 7,
     (4, 1): 1,
     (4, 2): 5,
     (4, 3): 9,
     (4, 4): 13,
     (5, 1): 1,
     (5, 2): 6,
     (5, 3): 12,
     (5, 4): 22,
     (6, 1): 1,
     (6, 2): 7,
     (6, 3): 16,
     (6, 4): 36,
     (7, 1): 1,
     (7, 2): 8,
     (7, 3): 20,
     (7, 4): 55,
     (8, 1): 1,
     (8, 2): 9,
     (8, 3): 25,
     (8, 4): 81,
     (9, 1): 1,
     (9, 2): 10,
     (9, 3): 30,
     (9, 4): 114,
     (10, 1): 1,
     (10, 2): 11,
     (10, 3): 36,
     (10, 4): 156}


def cal_num_B_features():
    """function to precalculate the number of B features"""
    n_B_feat_dict = {}

    for l_max in range(1,11):

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
