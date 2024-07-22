import h5py
import pickle
import torch
import numpy as np
import sys
import argparse

from ase.io import read
from cace.data import AtomicData

# Function to convert tensors to numpy arrays
def tensor_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, dict):
        return {k: tensor_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [tensor_to_numpy(item) for item in data]
    else:
        return data

# Function to convert numpy arrays back to tensors
def numpy_to_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, dict):
        return {k: numpy_to_tensor(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [numpy_to_tensor(item) for item in data]
    else:
        return data

# Function to save the dataset to an HDF5 file
def save_dataset(data, filename, shuffle=False):
    if shuffle:
        index = np.random.permutation(len(data)) # Shuffle the data
    else:
        index = np.arange(len(data))
    with h5py.File(filename, 'w') as f:
        for i, index_now in enumerate(index):
            item = data[index_now]
            grp = f.create_group(str(i))
            serializable_item = tensor_to_numpy(item)
            for k, v in serializable_item.items():
                grp.create_dataset(k, data=v)
    print(f"Saved dataset with {len(data)} records to {filename}")

# Function to read the dataset from an HDF5 file
def load_dataset(filename):
    all_data = []
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            grp = f[key]
            item = {k: grp[k][:] for k in grp.keys()}
            all_data.append(numpy_to_tensor(item))
    print(f"Loaded dataset with {len(all_data)} records from {filename}")
    return all_data

def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Split a dataset into multiple parts')
    parser.add_argument('--input_file', type=str, help='Path to the input dataset')
    parser.add_argument('--num_splits', type=int, help='Number of splits', default=4)
    parser.add_argument('--output_prefix', type=str, help='Prefix for the output files', default='split')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset before splitting')
    parser.add_argument('--cutoff', type=float, help='Cutoff radius for the atomic environment')
    parser.add_argument('--energy', type=str, help='Key for the energy data', default='energy')
    parser.add_argument('--forces', type=str, help='Key for the forces data', default='forces')
    parser.add_argument('--atomic_energies', type=str, help='file for the atomic energies', default=None)
    args = parser.parse_args()

    all_xyz_path = args.input_file
    num_splits = args.num_splits
    cutoff = args.cutoff
    data_key = {'energy': args.energy, 'forces': args.forces}
    atomic_energies = pickle.load(open(args.atomic_energies, 'rb')) if args.atomic_energies is not None else None

    for i in range(num_splits):
        all_xyz = read(all_xyz_path, index=slice(0, None, num_splits))

        dataset=[
            AtomicData.from_atoms(atoms, cutoff=cutoff, data_key=data_key, atomic_energies=atomic_energies).to_dict()  # Convert to dictionary
            for atoms in all_xyz 
            ]

        shuffle = args.shuffle  # Set to True if you want to shuffle the data
        save_dataset(dataset, args.output_prefix+'_'+str(i)+'.h5')

if __name__ == '__main__':
    main()

"""
load the dataset
# Load the dataset
loaded_dataset = load_dataset('test.h5')
print(loaded_dataset[0].keys())

dataset = [
    AtomicData(**data) # Convert to AtomicData object
    for data in loaded_dataset
    ]
"""
