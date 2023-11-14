from typing import Dict
import torch
from torch import nn

from .utils import get_outputs

__all__ = ['Forces']

class Forces(nn.Module):
    """
    Predicts forces and stress as response of the energy prediction

    """

    def __init__(
        self,
        training: bool = True,
        calc_forces: bool = True,
        calc_stress: bool = False,
        energy_key: str = 'CACE_energy',
        force_key: str = 'CACE_forces',
        stress_key: str = 'CACE_stress',
        virials_key: str = 'CACE_virials',
    ):
        """
        Args:
            calc_forces: If True, calculate atomic forces.
            calc_stress: If True, calculate the stress tensor.
            energy_key: Key of the energy in results.
            force_key: Key of the forces in results.
            stress_key: Key of the stress in results.
        """
        super().__init__()
        self.training = training
        self.calc_forces = calc_forces
        self.calc_stress = calc_stress
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key
        self.virial_key = virials_key
        self.model_outputs = []
        if calc_forces:
            self.model_outputs.append(force_key)
        if calc_stress:
            raise NotImplementedError("Stress is not implemented yet")
            self.model_outputs.append(stress_key)

        self.required_derivatives = []
        if self.calc_forces or self.calc_stress:
            self.required_derivatives.append('positions')

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        forces, virials, stress = get_outputs(
            energy=data.CACE_energy,
            positions=data['positions'],
            training=self.training,
            compute_force=self.calc_forces,
            compute_virials=self.calc_stress,
            compute_stress=self.calc_stress
            )

        data[self.force_key] = forces
        if self.virial_key is not None:
            data[self.virial_key] = virials
        if self.stress_key is not None:
            data[self.stress_key] = stress
        return data 
