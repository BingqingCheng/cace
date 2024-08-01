import torch
import torch.nn as nn
from typing import Dict, List

__all__ = ['CombinePotential']

class CombinePotential(nn.Module):
    def __init__(
        self,
        potentials: List[nn.Module],
        potential_keys: List[Dict],
        out_keys: List = ['CACE_energy', 'CACE_forces', 'CACE_stress'],
        operation = None,
    ):
        """
        Combine multiple potentials into a single potential.
        Args:
        potentials: List of potentials to combine.
        potential_keys: List of dictionaries with keys for each potential.
                       e.g. [pot1, pot2] where
        pot1 = {'CACE_energy': 'CACE_energy_intra',
        'CACE_forces': 'CACE_forces_intra',
        }

        pot2 = {'CACE_energy': 'CACE_energy_inter',
        'CACE_forces': 'CACE_forces_inter',
        }
        out_keys: List of keys to output. Should be present in all potential_keys.
        operation: Operation to combine the outputs of the potentials.
        """
        super().__init__()
        self.models =  nn.ModuleList([potential for potential in potentials])
        self.potential_keys = potential_keys
        self.required_derivatives = []
        for potential in potentials:
            for d in potential.required_derivatives:
                if d not in self.required_derivatives:
                    self.required_derivatives.append(d)

        self.out_keys = []
        for key in out_keys:
            if all(key in potential_key for potential_key in self.potential_keys):
                self.out_keys.append(key)

        if operation is None:
            # Default operation (sum)
            self.operation = self.default_operation
        else:
            self.operation = operation

    def default_operation(self, my_list):
        return torch.stack(my_list).sum(0)


    def forward(self,
                data: Dict[str, torch.Tensor],
                training: bool = False,
                compute_stress: bool = False,
                compute_virials: bool = False,
                output_index: int = None, # only used for multiple-head output
                ) -> Dict[str, torch.Tensor]:
        results = {}
        output = {}
        for i, potential in enumerate(self.models):
            result = potential(data, training, compute_stress, compute_virials, output_index)
            results[i] = result
            output.update(result)

        for key in self.out_keys:
            values = [ results[i][potential_key[key]] for i, potential_key in enumerate(self.potential_keys)]
            if values:
                output[key] = self.operation(values)
        return output
