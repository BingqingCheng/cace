# the CACE calculator for ASE

from typing import Union, List

import numpy as np 
import torch

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from ..tools import torch_geometric, torch_tools, to_numpy
from ..data import AtomicData
 
__all__ = ["CACECalculator"]

class CACECalculator(Calculator):
    """CACE ASE Calculator
    args:
        model_path: str or nn.module, path to model
        device: str, device to run on (cuda or cpu)
        compute_stress: bool, whether to compute stress
        energy_key: str, key for energy in model output
        forces_key: str, key for forces in model output
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        atomic_energies: dict, dictionary of atomic energies to add to model output
    """

    def __init__(
        self,
        model_path: Union[str, torch.nn.Module],
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        electric_field_unit: float = 1.0,
        compute_stress = False,
        energy_key: str = 'energy',
        forces_key: str = 'forces',
        stress_key: str = 'stress',
        bec_key: str = 'bec',
        external_field: Union[float,List[float]] = None,
        atomic_energies: dict = None,
        output_index: int = None, # only used for multi-output models
        **kwargs,
        ):

        Calculator.__init__(self, **kwargs)
        self.implemented_properties = [
            "energy",
            "forces",
            "stress",
        ]

        self.results = {}

        if isinstance(model_path, str):
            self.model = torch.load(f=model_path, map_location=device)
        elif isinstance(model_path, torch.nn.Module):
            self.model = model_path
        else:
            raise ValueError("model_path must be a string or nn.Module")
        self.model.to(device)

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.electric_field_unit = electric_field_unit

        try:
            self.cutoff = self.model.representation.cutoff
        except AttributeError:
            self.cutoff = self.model.models[0].representation.cutoff

        self.atomic_energies = atomic_energies

        self.compute_stress = compute_stress
        self.energy_key = energy_key 
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.bec_key = bec_key

        if external_field is not None:
            if isinstance(external_field, float):
                self.external_field = external_field
            else:
                self.external_field = np.array(external_field)
        else:
            self.external_field = None      
        self.output_index = output_index
        
        for param in self.model.parameters():
            param.requires_grad = False

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        if not hasattr(self, "output_index"):
            self.output_index = None

        # prepare data
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                AtomicData.from_atoms(
                    atoms, cutoff=self.cutoff
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        batch_base = next(iter(data_loader)).to(self.device)
        batch = batch_base.clone()
        output = self.model(batch.to_dict(), training=True, compute_stress=self.compute_stress, output_index=self.output_index)
        energy_output = to_numpy(output[self.energy_key])
        forces_output = to_numpy(output[self.forces_key])
        if self.external_field is not None and self.bec_key is not None:
            bec_output = to_numpy(output[self.bec_key])
        # subtract atomic energies if available
        if self.atomic_energies:
            e0 = sum(self.atomic_energies.get(Z, 0) for Z in atoms.get_atomic_numbers())
        else:
            e0 = 0.0
        self.results["energy"] = (energy_output + e0) * self.energy_units_to_eV
        self.results["forces"] = forces_output * self.energy_units_to_eV / self.length_units_to_A
        if self.external_field is not None:
            if isinstance(self.external_field, float):
                self.results["forces"] += bec_output * self.external_field * self.electric_field_unit
            else:
                self.results["forces"] += bec_output @ self.external_field * self.electric_field_unit
        if self.compute_stress and output[self.stress_key] is not None:
            stress = to_numpy(output[self.stress_key])
            # stress has units eng / len^3:
            self.results["stress"] = (
                stress * (self.energy_units_to_eV / self.length_units_to_A**3)
            )[0]
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])

        return self.results
