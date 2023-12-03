# the CACE calculator for ASE

import numpy as np 
import torch

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from ..tools import torch_geometric, torch_tools, to_numpy
from ..data import AtomicData, config_from_atoms
 
__all__ = ["CACECalculator"]

class CACECalculator(Calculator):
    """CACE ASE Calculator
    args:
        model_path: str, path to model
        device: str, device to run on (cuda or cpu)
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        **kwargs,
        ):

        Calculator.__init__(self, **kwargs)
        self.implemented_properties = [
            "energy",
            "forces",
            "stress",
        ]

        self.results = {}

        self.model = torch.load(f=model_path, map_location=device)
        self.model.to(device)

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A

        self.cutoff = self.model.representation.cutoff

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

        # prepare data
        config = config_from_atoms(atoms)
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

        compute_stress = False
        batch_base = next(iter(data_loader)).to(self.device)
        batch = batch_base.clone()
        out = self.model(batch.to_dict(), compute_stress=compute_stress)
        self.results["energy"] = to_numpy(out["CACE_energy"]) * self.energy_units_to_eV
        self.results["forces"] = to_numpy(out["CACE_forces"]) * self.energy_units_to_eV / self.length_units_to_A
        if compute_stress and out["stress"] is not None:
            self.results["stress"] = full_3x3_to_voigt_6_stress(to_numpy(out["CACE_stress"])) * self.energy_units_to_eV / self.length_units_to_A**3

        return self.results
