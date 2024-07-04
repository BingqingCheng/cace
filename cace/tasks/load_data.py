import dataclasses
import logging
from typing import Dict, List, Optional, Tuple, Sequence
import numpy as np
from ase import Atoms
from ase.io import read
from ..tools import torch_geometric
from ..data import AtomicData

__all__ = ["load_data_loader", "get_dataset_from_xyz", "random_train_valid_split"]

@dataclasses.dataclass
class SubsetAtoms:
    train: Atoms
    valid: Atoms 
    test: Atoms
    cutoff: float
    data_key: Dict
    atomic_energies: Dict 

def load_data_loader(
    collection: SubsetAtoms,
    data_type: str, # ['train', 'valid', 'test']
    batch_size: int,
):

    allowed_types = ['train', 'valid', 'test']
    if data_type not in allowed_types:
        raise ValueError(f"Input value must be one of {allowed_types}, got {data_type}")

    cutoff = collection.cutoff
    data_key = collection.data_key
    atomic_energies = collection.atomic_energies

    if data_type == 'train':
        loader = torch_geometric.DataLoader(
            dataset=[
                AtomicData.from_atoms(atoms, cutoff=cutoff, data_key=data_key, atomic_energies=atomic_energies)
                for atoms in collection.train
            ],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    elif data_type == 'valid':
        loader = torch_geometric.DataLoader(
            dataset=[
                AtomicData.from_atoms(atoms, cutoff=cutoff, data_key=data_key, atomic_energies=atomic_energies)
                for atoms in collection.valid
            ],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
    elif data_type == 'test':
        loader = torch_geometric.DataLoader(
            dataset=[
                AtomicData.from_atoms(atoms, cutoff=cutoff, data_key=data_key, atomic_energies=atomic_energies)
                for atoms in collection.test
            ],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
    return loader

def get_dataset_from_xyz(
    train_path: str,
    cutoff: float,
    valid_path: str = None,
    valid_fraction: float = 0.1,
    test_path: str = None,
    seed: int = 1234,
    data_key: Dict[str, str] = None,
    atomic_energies: Dict[int, float] = None
) -> SubsetAtoms:
    """Load training and test dataset from xyz file"""
    all_train_configs = read(train_path, ":")
    if not isinstance(all_train_configs, list):
        all_train_configs = [all_train_configs]
    logging.info(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )
    if valid_path is not None:
        valid_configs = read(valid_path, ":")
        if not isinstance(valid_configs, list):
            valid_configs = [valid_configs]
        logging.info(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
        )
        train_configs = all_train_configs
    else:
        logging.info(
            "Using random %s%% of training set for validation", 100 * valid_fraction
        )
        train_configs, valid_configs = random_train_valid_split(
            all_train_configs, valid_fraction, seed
        )

    test_configs = []
    if test_path is not None:
        test_configs = read(test_path, ":")
        if not isinstance(test_configs, list):
            test_configs = [test_configs]
        logging.info(
            f"Loaded {len(test_configs)} test configurations from '{test_path}'"
        )
    return (
        SubsetAtoms(train=train_configs, valid=valid_configs, test=test_configs, cutoff=cutoff, data_key=data_key, atomic_energies=atomic_energies)
    )

def random_train_valid_split(
    items: Sequence, valid_fraction: float, seed: int
) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    train_size = size - int(valid_fraction * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )
