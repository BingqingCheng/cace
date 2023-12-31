import dataclasses
import logging
from typing import Dict, List, Optional, Tuple

from ..data import Configurations, load_from_xyz, random_train_valid_split
from ..tools import torch_geometric
from ..data import AtomicData

__all__ = ["load_data_loader", "get_dataset_from_xyz"]

@dataclasses.dataclass
class SubsetCollection:
    train: Configurations
    valid: Configurations
    test: List[Tuple[str, Configurations]]


def load_data_loader(
    collection: SubsetCollection,
    data_type: str, # ['train', 'valid', 'test']
    batch_size: int,
    cutoff: float
):
    allowed_types = ['train', 'valid', 'test']
    if data_type not in allowed_types:
        raise ValueError(f"Input value must be one of {allowed_types}, got {data_type}")

    if data_type == 'train':
        loader = torch_geometric.DataLoader(
            dataset=[
                AtomicData.from_config(config, cutoff=cutoff)
                for config in collection.train
            ],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    elif data_type == 'valid':
        loader = torch_geometric.DataLoader(
            dataset=[
                AtomicData.from_config(config, cutoff=cutoff)
                for config in collection.valid
            ],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
    elif data_type == 'test':
        loader = torch_geometric.DataLoader(
            dataset=[
                AtomicData.from_config(config, cutoff=cutoff)
                for config in collection.test
            ],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
    return loader

def get_dataset_from_xyz(
    train_path: str,
    valid_path: str = None,
    valid_fraction: float = 0.1,
    config_type_weights: Dict[str, float] = None,
    test_path: str = None,
    seed: int = 1234,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipoles",
    charges_key: str = "charges",
    atomic_energies: Dict[int, float] = None
) -> SubsetCollection:
    """Load training and test dataset from xyz file"""
    all_train_configs = load_from_xyz(
        file_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        atomic_energies=atomic_energies,
    )
    logging.info(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )
    if valid_path is not None:
        valid_configs = load_from_xyz(
            file_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            atomic_energies=atomic_energies,
        )
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
        test_configs = load_from_xyz(
            file_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            atomic_energies=atomic_energies,
        )
        logging.info(
            f"Loaded {len(test_configs)} test configurations from '{test_path}'"
        )
    return (
        SubsetCollection(train=train_configs, valid=valid_configs, test=test_configs)
    )
