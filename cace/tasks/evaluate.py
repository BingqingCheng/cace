import numpy as np 
import torch
from torch import nn

from ase import Atoms

from ..tools import torch_geometric, torch_tools, to_numpy
from ..data import AtomicData, config_from_atoms

__all__ = ["EvaluateTask"]

class EvaluateTask(nn.Module):
    """CACE Evaluator 
    args:
        model_path: str, path to model
        device: str, device to run on (cuda or cpu)
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        energy_key: str = 'energy',
        forces_key: str = 'forces',
        stress_key: str = 'stress',
        ):

        super().__init__()

        self.model = torch.load(f=model_path, map_location=device)
        self.model.to(device)

        self.device = torch_tools.init_device(device)
        self.cutoff = self.model.representation.cutoff
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key
        
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, data=None, batch_size=1, compute_stress=False):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :return:
        """
        # Collect data
        energies_list = []
        stresses_list = []
        forces_list = []

        # check the data type
        if isinstance(data, torch_geometric.batch.Batch):
            data.to(self.device)
            output = self.model(data.to_dict())
            energies_list.append(to_numpy(output[self.energy_key]))
            forces = np.split(
                torch_tools.to_numpy(output[self.forces_key]),
                indices_or_sections=data.ptr[1:],
                axis=0,
            )
            forces_list.append(forces[:-1])
            if compute_stress and self.stress_key in output:
                stresses_list.append(to_numpy(output[self.stress_key]))

        elif isinstance(data, Atoms):
            config = config_from_atoms(data)
            data_loader = torch_geometric.dataloader.DataLoader(
                dataset=[
                        AtomicData.from_config(
                        config, cutoff=self.cutoff
                        )
                ],
                batch_size=1,
                shuffle=False,
                drop_last=False,
            )
            output = self.model(next(iter(data_loader)).to_dict())
            energies_list.append(to_numpy(output[self.energy_key]))
            forces_list.append(to_numpy(output[self.forces_key]))
            if compute_stress and self.stress_key in output:
                stresses_list.append(to_numpy(output[self.stress_key]))

        # check if the data is a list of atoms
        elif isinstance(data, list):
            if not isinstance(data[0], Atoms):
               raise ValueError("Input data must be a list of ASE Atoms objects")
            configs = [config_from_atoms(atoms) for atoms in data]
            data_loader = torch_geometric.dataloader.DataLoader(
                dataset=[
		    AtomicData.from_config(
			config, cutoff=self.cutoff
		    )
		    for config in configs
		],
		batch_size=batch_size,
		shuffle=False,
		drop_last=False,
	    )
            for batch in data_loader:
                batch.to(self.device)
                output = self.model(batch.to_dict())
                energies_list.append(to_numpy(output[self.energy_key]))
                forces = np.split(
                    to_numpy(output[self.forces_key]),
                    indices_or_sections=batch.ptr[1:],
                    axis=0,
                )
                forces_list.append(forces[:-1])
            if compute_stress and self.stress_key in output:
                stresses_list.append(to_numpy(output[self.stress_key]))

        elif isinstance(data, torch_geometric.dataloader.DataLoader):
            for batch in data:
                batch.to(self.device)
                output = self.model(batch.to_dict())
                energies_list.append(to_numpy(output[self.energy_key]))
                forces = np.split(
                    torch_tools.to_numpy(output[self.forces_key]),
                    indices_or_sections=batch.ptr[1:],
                    axis=0,
                )
                forces_list.append(forces[:-1])
            if compute_stress and self.stress_key in output:
                stresses_list.append(to_numpy(output[self.stress_key]))
        else:
            raise ValueError("Input data type not recognized")

        results = {
            "energy": np.concatenate(energies_list) * self.energy_units_to_eV,
            "forces": np.concatenate(forces_list) * self.energy_units_to_eV / self.length_units_to_A,
            "stress": None if len(stresses_list) == 0 else np.concatenate(stresses_list) * self.energy_units_to_eV / self.length_units_to_A ** 3,
	}
        return results
