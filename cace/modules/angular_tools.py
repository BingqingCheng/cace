import itertools
from collections import defaultdict
from itertools import permutations
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

def combine_and_sum(entries):
    combined_entries = defaultdict(int)
    result = []
    for entry in entries:
        prefactor = entry[-1]
        sorted_vectors = tuple(sorted(map(tuple, entry[:-1])))
        combined_entries[sorted_vectors] += prefactor
    for vectors, prefactor in combined_entries.items():
        result.append((*map(list, vectors), prefactor))
    return result

def gen_lxlylz_combo(l_max, remove_zero=False):
    all_lxlylz = []
    if l_max <= 0:
        return all_lxlylz
    for lx, ly, lz in itertools.product(range(l_max+1), repeat=3):
        l = lx + ly + lz
        if l <= l_max:
           if remove_zero and l == 0:
               continue
           all_lxlylz.append([lx, ly, lz, l])
    return all_lxlylz

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
    
    for L in range(1, l_max+1):
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

    for dl1 in range(1, l_max):
        dl2_max = min(dl1, l_max - dl1)
        for dl2 in range(1, dl2_max+1):
            dl3_max = min(dl2, l_max - dl1, l_max - dl2)
            for dl3 in range(0, dl3_max+1):
                for dlxlylz1 in make_lxlylz(dl1):
                    for dlxlylz2 in make_lxlylz(dl2):
                        for dlxlylz3 in make_lxlylz(dl3):
                            lxlylz1 = [sum(x) for x in zip(dlxlylz1, dlxlylz2)]
                            lxlylz2 = [sum(x) for x in zip(dlxlylz1, dlxlylz3)]
                            lxlylz3 = [sum(x) for x in zip(dlxlylz2, dlxlylz3)]
                            prefactor = lxlylz_factorial_coef(dlxlylz1)*lxlylz_factorial_coef(dlxlylz2)*lxlylz_factorial_coef(dlxlylz3)
                            #key = (dl1 + dl2, dl1 + dl3, dl2 + dl3)
                            key = (dl1, dl2, dl3)
                            vec_dict[key] = vec_dict.get(key, []) + [(lxlylz1, lxlylz2, lxlylz3, prefactor)]

    for key in vec_dict:
        vec_dict[key] = combine_and_sum(vec_dict[key])

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
    for dl1 in range(1, l_max):
        dl2_max = min(dl1, l_max - dl1)
        for dl2 in range(1, dl2_max+1):
            dl3_max = min(dl2, l_max - dl2)
            for dl3 in range(1, dl3_max+1):
                dl4_max = min(dl3, l_max - dl1)
                for dl4 in range(0, dl4_max+1):
                    cl1_max = min(l_max - dl1 - dl4, l_max - dl2 - dl3)
                    cl2_max = min(l_max - dl1 - dl2, l_max - dl2 - dl4)
                    for cl1 in range(0, cl1_max+1):
                        for cl2 in range(0, cl2_max+1):

                            for dlxlylz1 in make_lxlylz(dl1):
                                for dlxlylz2 in make_lxlylz(dl2):
                                    for dlxlylz3 in make_lxlylz(dl3):
                                        for dlxlylz4 in make_lxlylz(dl4):
                                            for clxlylz1 in make_lxlylz(cl1):
                                                for clxlylz2 in make_lxlylz(cl2):
                                                    lxlylz1 = [sum(x) for x in zip(dlxlylz4, dlxlylz1, clxlylz1)]
                                                    lxlylz2 = [sum(x) for x in zip(dlxlylz1, dlxlylz2, clxlylz2)]
                                                    lxlylz3 = [sum(x) for x in zip(dlxlylz2, dlxlylz3, clxlylz1)]
                                                    lxlylz4 = [sum(x) for x in zip(dlxlylz3, dlxlylz4, clxlylz2)]
                                                    prefactor = lxlylz_factorial_coef(dlxlylz1)*lxlylz_factorial_coef(dlxlylz2)*lxlylz_factorial_coef(dlxlylz3)*lxlylz_factorial_coef(dlxlylz4)*lxlylz_factorial_coef(clxlylz1)*lxlylz_factorial_coef(clxlylz2)
                                                    #l1, l2, l3, l4 = sorted((sum(lxlylz1), sum(lxlylz2), sum(lxlylz3), sum(lxlylz4)))
                                                    #if l1==l2 and l3==l4: continue
                                                    #key = (l1, l2, l3, l4)
                                                    key = (dl1, dl2, dl3, dl4, cl1, cl2)
                                                    vec_dict[key] = vec_dict.get(key, []) + [(lxlylz1, lxlylz2, lxlylz3, lxlylz4,  prefactor)]

    for key in vec_dict:
        vec_dict[key] = combine_and_sum(vec_dict[key])

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
