###########################################################################################
# Atomic Data Class for handling molecules as graphs
# modified from MACE
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Optional, Sequence

#import torch_geometric
from ..tools import torch_geometric
import torch.nn as nn
import torch.utils.data
from ..tools import voigt_to_matrix

from .neighborhood import get_neighborhood
from .utils import Configuration


class AtomicData(torch_geometric.data.Data):
    atomic_numbers: torch.Tensor
    num_graphs: torch.Tensor
    num_nodes: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    n_atom_basis: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor
    virials: torch.Tensor
    dipole: torch.Tensor
    charges: torch.Tensor
    weight: torch.Tensor
    energy_weight: torch.Tensor
    forces_weight: torch.Tensor
    stress_weight: torch.Tensor
    virials_weight: torch.Tensor

    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges], always sender -> receiver
        atomic_numbers: torch.Tensor,  # [n_nodes]
        num_nodes: torch.Tensor, #[,]
        positions: torch.Tensor,  # [n_nodes, 3]
        shifts: torch.Tensor,  # [n_edges, 3],
        unit_shifts: torch.Tensor,  # [n_edges, 3]
        cell: Optional[torch.Tensor],  # [3,3]
        weight: Optional[torch.Tensor],  # [,]
        energy_weight: Optional[torch.Tensor],  # [,]
        forces_weight: Optional[torch.Tensor],  # [,]
        stress_weight: Optional[torch.Tensor],  # [,]
        virials_weight: Optional[torch.Tensor],  # [,]
        forces: Optional[torch.Tensor],  # [n_nodes, 3]
        energy: Optional[torch.Tensor],  # [, ]
        stress: Optional[torch.Tensor],  # [1,3,3]
        virials: Optional[torch.Tensor],  # [1,3,3]
        dipole: Optional[torch.Tensor],  # [, 3]
        charges: Optional[torch.Tensor],  # [n_nodes, ]
    ):
        # Check shapes
        #assert num_nodes == atomic_numbers.shape[0]
        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert weight is None or len(weight.shape) == 0
        assert energy_weight is None or len(energy_weight.shape) == 0
        assert forces_weight is None or len(forces_weight.shape) == 0
        assert stress_weight is None or len(stress_weight.shape) == 0
        assert virials_weight is None or len(virials_weight.shape) == 0
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape == (num_nodes, 3)
        assert energy is None or len(energy.shape) == 0
        assert stress is None or stress.shape == (1, 3, 3)
        assert virials is None or virials.shape == (1, 3, 3)
        assert dipole is None or dipole.shape[-1] == 3
        assert charges is None or charges.shape == (num_nodes,)
        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "atomic_numbers": atomic_numbers,
            "num_nodes": num_nodes,
            "weight": weight,
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
            "stress_weight": stress_weight,
            "virials_weight": virials_weight,
            "forces": forces,
            "energy": energy,
            "stress": stress,
            "virials": virials,
            "dipole": dipole,
            "charges": charges,
        }
        super().__init__(**data)

    @classmethod
    def from_config(
        cls, config: Configuration, 
        cutoff: float, 
    ) -> "AtomicData":
        edge_index, shifts, unit_shifts  = get_neighborhood(
            positions=config.positions, cutoff=cutoff, pbc=config.pbc, cell=config.cell
        )
  
        atomic_numbers = torch.tensor(config.atomic_numbers, dtype=torch.long)

        cell = (
            torch.tensor(config.cell, dtype=torch.get_default_dtype())
            if config.cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        weight = (
            torch.tensor(config.weight, dtype=torch.get_default_dtype())
            if config.weight is not None
            else 1
        )

        energy_weight = (
            torch.tensor(config.energy_weight, dtype=torch.get_default_dtype())
            if config.energy_weight is not None
            else 1
        )

        forces_weight = (
            torch.tensor(config.forces_weight, dtype=torch.get_default_dtype())
            if config.forces_weight is not None
            else 1
        )

        stress_weight = (
            torch.tensor(config.stress_weight, dtype=torch.get_default_dtype())
            if config.stress_weight is not None
            else 1
        )

        virials_weight = (
            torch.tensor(config.virials_weight, dtype=torch.get_default_dtype())
            if config.virials_weight is not None
            else 1
        )

        forces = (
            torch.tensor(config.forces, dtype=torch.get_default_dtype())
            if config.forces is not None
            else None
        )
        energy = (
            torch.tensor(config.energy, dtype=torch.get_default_dtype())
            if config.energy is not None
            else None
        )
        stress = (
            voigt_to_matrix(
                torch.tensor(config.stress, dtype=torch.get_default_dtype())
            ).unsqueeze(0)
            if config.stress is not None
            else None
        )
        virials = (
            torch.tensor(config.virials, dtype=torch.get_default_dtype()).unsqueeze(0)
            if config.virials is not None
            else None
        )
        dipole = (
            torch.tensor(config.dipole, dtype=torch.get_default_dtype()).unsqueeze(0)
            if config.dipole is not None
            else None
        )
        charges = (
            torch.tensor(config.charges, dtype=torch.get_default_dtype())
            if config.charges is not None
            else None
        )

        return cls(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=torch.tensor(config.positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
            cell=cell,
            atomic_numbers=atomic_numbers,
            num_nodes=atomic_numbers.shape[0],
            weight=weight,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            stress_weight=stress_weight,
            virials_weight=virials_weight,
            forces=forces,
            energy=energy,
            stress=stress,
            virials=virials,
            dipole=dipole,
            charges=charges,
        )


def get_data_loader(
    dataset: Sequence[AtomicData],
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
