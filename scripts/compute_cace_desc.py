#import sys
#sys.path.append('../')

import os
from tqdm import tqdm
import numpy as np
import torch
import ase
from ase import Atoms
from ase.io import read,write

import cace
from cace.data import AtomicData
from cace.representations.cace_representation import Cace
from cace.tools import to_numpy
from cace.tools import scatter_sum

cutoff = 4.0
batch_size = 10 


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, 'cace_alchemical_rep_v0.pth')
cace_repr = torch.load(data_file_path, map_location='cpu')

data = read(sys.argv[1], ":")
dataset=[
    AtomicData.from_atoms(
    atom, cutoff=cutoff
    )
    for atom in data
    ]

data_loader = cace.tools.torch_geometric.dataloader.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

n_frame = 0
for sampled_data in tqdm(data_loader):
    cace_result_more = cace_repr(sampled_data)
    avg_B = scatter_sum(
        src=cace_result_more['node_feats'], 
        index=sampled_data["batch"], 
        dim=0
        )
    n_configs = avg_B.shape[0]
    avg_B_flat = to_numpy(avg_B.reshape(n_configs, -1))
    for i in range(n_configs):
        data[i+n_frame].info['CACE_desc'] = avg_B_flat[i]
    n_frame += n_configs

# check if sys.argv[2] exists
if len(sys.argv) > 2:
    prefix = sys.argv[2]
else:
    prefix = 'CACE_desc'
write(prefix+'.xyz', data)






