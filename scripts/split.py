import h5py
import pickle
import torch
import numpy as np
import sys
import argparse

from ase.io import read
from cace.data import AtomicData
from cace.tools import save_dataset

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
        all_xyz = read(all_xyz_path, index=slice(i, None, num_splits))

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
