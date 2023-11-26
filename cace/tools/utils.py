import json
import logging
import os
import sys
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np
import torch

from .torch_tools import to_numpy

class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))


def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)

def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader) -> float:
    num_neighbors = []

    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()

def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
):
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (directory is not None) and (tag is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, tag + ".log")
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)

        logger.addHandler(fh)
