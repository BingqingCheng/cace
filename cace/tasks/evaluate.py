from typing import Union
import numpy as np 
import torch
from torch import nn

from ase import Atoms
from ase.io import read, write

from ..tools import torch_geometric, torch_tools, to_numpy
from ..data import AtomicData

__all__ = ["EvaluateTask"]

class EvaluateTask(nn.Module):
    """CACE Evaluator 
    args:
        model_path: str, path to model
        device: str, device to run on (cuda or cpu)
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        energy_key: str, name of energy key in model output
        forces_key: str, name of forces key in model output
        stress_key: str, name of stress key in model output
        atomic_energies: dict, dictionary of atomic energies to add to model output
    """

    def __init__(
        self,
        model_path: Union[str, nn.Module],
        device: str = "cpu",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        energy_key: str = 'energy',
        forces_key: str = 'forces',
        stress_key: str = 'stress',
        other_keys: list = [],
        atomic_energies: dict = None,
        data_key: dict = None,
        ):

        super().__init__()

        if isinstance(model_path, str):
            self.model = torch.load(f=model_path, map_location=device)
        elif isinstance(model_path, nn.Module):
            self.model = model_path
        else:
            raise ValueError("model_path must be a string or nn.Module")

        self.model.to(device)

        self.device = torch_tools.init_device(device)
        try:
            self.cutoff = self.model.representation.cutoff
        except AttributeError:
            self.cutoff = self.model.models[0].representation.cutoff
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.other_keys = other_keys
        self.data_key = data_key

        self.atomic_energies = atomic_energies
        
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, data=None, batch_size=1, compute_stress=False, xyz_output=None):
        """
        Calculate properties.
        args:
             data: torch_geometric.data.Data, torch_geometric.data.Batch, list of ASE Atoms objects, or torch_geometric.data.DataLoader
             batch_size: int, batch size
             compute_stress: bool, whether to compute stress
        """
        # Collect data
        energies_list = []
        stresses_list = []
        forces_list = []
        other_outputs = {key: [] for key in self.other_keys}

        # check the data type
        if isinstance(data, torch_geometric.batch.Batch):
            data.to(self.device)
            output = self.model(data.to_dict())
            if self.energy_key in output:
                energies_now = to_numpy(output[self.energy_key])
                if self.atomic_energies is not None:
                    e0_list = self._add_atomic_energies(data)
                    if len(energies_now.shape) > 1:
                        n_entry = energies_now.shape[1]
                        e0_list = np.repeat(e0_list, n_entry).reshape(-1, n_entry) 
                        energies_list.append(energies_now + e0_list)
                    else:
                        energies_list.append(energies_now)
            if self.forces_key in output:
                forces_list.append(to_numpy(output[self.forces_key]))
            if compute_stress and self.stress_key in output:
                stresses_list.append(to_numpy(output[self.stress_key]))
            for key in self.other_keys:
                if key in output:
                    other_outputs[key].append(to_numpy(output[key]))

        elif isinstance(data, Atoms):
            data_loader = torch_geometric.dataloader.DataLoader(
                dataset=[
                        AtomicData.from_atoms(
                        data, cutoff=self.cutoff,
                        data_key=self.data_key
                        )
                ],
                batch_size=1,
                shuffle=False,
                drop_last=False,
            )
            output = self.model(next(iter(data_loader)).to_dict())
            if self.energy_key in output:
                energy = to_numpy(output[self.energy_key])
                if self.atomic_energies is not None:
                    atomic_numbers = data.get_atomic_numbers()
                    energy += sum(self.atomic_energies.get(Z, 0) for Z in atomic_numbers)
                energies_list.append(energy)
            if self.forces_key in output:
                forces_list.append(to_numpy(output[self.forces_key]))
            if compute_stress and self.stress_key in output:
                stresses_list.append(to_numpy(output[self.stress_key]))
            for key in self.other_keys:
                if key in output:
                    other_outputs[key].append(to_numpy(output[key]))

        # check if the data is a list of atoms
        elif isinstance(data, list):
            if not isinstance(data[0], Atoms):
               raise ValueError("Input data must be a list of ASE Atoms objects")
            data_loader = torch_geometric.dataloader.DataLoader(
                dataset=[
		    AtomicData.from_atoms(
			atom, cutoff=self.cutoff,
                        data_key=self.data_key
		    )
		    for atom in data
		],
		batch_size=batch_size,
		shuffle=False,
		drop_last=False,
	    )
            atomforces_list = []
            for batch in data_loader:
                batch.to(self.device)
                output = self.model(batch.to_dict())
                if self.energy_key in output:
                    energies_now = to_numpy(output[self.energy_key])
                    if self.atomic_energies is not None:
                        e0_list = self._add_atomic_energies(batch)
                        if len(energies_now.shape) > 1:
                            n_entry = energies_now.shape[1]
                            e0_list = np.repeat(e0_list, n_entry).reshape(-1, n_entry) 
                        energies_now += e0_list
                    energies_list.append(energies_now)

                if self.forces_key in output:
                    forces_list.append(to_numpy(output[self.forces_key]))
                    forces = np.split(
                        to_numpy(output[self.forces_key]),
                        indices_or_sections=batch.ptr[1:],
                        axis=0,
                    )
                    atomforces_list.append(forces[:-1])
                if compute_stress and self.stress_key in output:
                    stresses_list.append(to_numpy(output[self.stress_key]))
                for key in self.other_keys:
                    if key in output:
                        other_outputs[key].append(to_numpy(output[key]))

            if xyz_output is not None and batch_size > 1:
                raise ValueError("Batch size must be 1 to write xyz files")

            atoms_list = []
            # Store data in atoms objects
            if xyz_output is not None and batch_size == 1:
                for i in range(len(data)):
                    atoms = data[i].copy()
                    atoms.calc = None  # crucial
                    if len(energies_list) >= 1:
                        atoms.info[self.energy_key] = energies_list[i][0] * self.energy_units_to_eV
                    if len(forces_list) >= 1:
                        atoms.set_array(self.forces_key, forces_list[i] * self.energy_units_to_eV / self.length_units_to_A)
                    for key in self.other_keys:
                        output_now = other_outputs[key][i]
                        #print(key, output_now.shape) 
                        if output_now.ndim > 2 and output_now.shape[0] == 1:
                            output_now = output_now[0]
                        if output_now.ndim > 2:
                            output_now = output_now.reshape(output_now.shape[0], -1)
                        #print(key, output_now.shape) 
                        # is complex
                        if np.iscomplexobj(output_now):
                            if output_now.shape[0] == len(atoms.get_positions()):
                                atoms.set_array(key+'_real', output_now.real)
                                atoms.set_array(key+'_imag', output_now.imag)
                            else:
                                atoms.info[key+'_real'] = output_now.real
                                atoms.info[key+'_imag'] = output_now.imag
                        else:
                            if output_now.shape[0] == len(atoms.get_positions()):
                                atoms.set_array(key, output_now)
                            else:
                                atoms.info[key] = output_now
                    if compute_stress:
                        atoms.info[self.stress_key] = stresses_list[i]
                    atoms_list.append(atoms)
      	 	    # Write atoms to output path
                    write(xyz_output, atoms, format="extxyz", append=True)
            #return atoms_list

        elif isinstance(data, torch_geometric.dataloader.DataLoader):
            for batch in data:
                batch.to(self.device)
                output = self.model(batch.to_dict())
                if self.energy_key in output:
                    energies_now = to_numpy(output[self.energy_key])
                    if self.atomic_energies is not None:
                        e0_list = self._add_atomic_energies(batch)
                        if len(energies_now.shape) > 1:
                            n_entry = energies_now.shape[1]
                            e0_list = np.repeat(e0_list, n_entry).reshape(-1, n_entry) 
                        energies_list.append(energies_now + e0_list)
                    else:
                        energies_list.append(energies_now)

                if self.forces_key in output:
                    forces_list.append(to_numpy(output[self.forces_key]))
                if compute_stress and self.stress_key in output:
                    stresses_list.append(to_numpy(output[self.stress_key]))
                for key in self.other_keys:
                    if key in output:
                        other_outputs[key].append(to_numpy(output[key]))
        else:
            raise ValueError("Input data type not recognized")

        results = {
            "energy": None if len(energies_list) == 0 else np.concatenate(energies_list) * self.energy_units_to_eV,
            "forces": None if len(forces_list) == 0 else np.vstack(forces_list) * self.energy_units_to_eV / self.length_units_to_A,
            "stress": None if len(stresses_list) == 0 else np.concatenate(stresses_list) * self.energy_units_to_eV / self.length_units_to_A ** 3,
	}
        for key in self.other_keys:
            results[key] = np.concatenate(other_outputs[key])
        return results

    def _add_atomic_energies(self, batch: torch_geometric.batch.Batch):
        e0_list = []
        atomic_numbers_list = np.split(to_numpy(batch['atomic_numbers']),
             indices_or_sections=batch.ptr[1:],
             axis=0,
             )[:-1]
        for atomic_numbers in atomic_numbers_list:
            e0_list.append(sum(self.atomic_energies.get(Z, 0) for Z in atomic_numbers))
        return np.array(e0_list)
