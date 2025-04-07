import torch
import torch.nn as nn
from typing import Dict, List, Optional

__all__ = ['CombinePotential']

# @torch.jit.script
def default_operation(my_list: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(my_list).sum(0)

class CombinePotential(nn.Module):
    def __init__(
        self,
        potentials: List[nn.Module],
        potential_keys: List[Dict],
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
        'weight': 1.
        }

        pot2 = {'CACE_energy': 'CACE_energy_inter',
        'CACE_forces': 'CACE_forces_inter',
        'weight': 0.01,
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
        for key in potential_keys[0]:
            if all(key in potential_key for potential_key in self.potential_keys) and key != 'weight':
                self.out_keys.append(key)

        if operation is None:
            # Default operation (sum)
            self.operation = default_operation
        else:
            self.operation = operation


    def forward(self,
                data: Dict[str, torch.Tensor],
                training: bool = False,
                compute_stress: bool = False,
                compute_virials: bool = False,
                output_index: Optional[int] = None, # only used for multiple-head output
                ) -> Dict[str, torch.Tensor]:
        results: Dict[str, Dict[str, torch.Tensor]] = torch.jit.annotate(
            Dict[str, Dict[str, torch.Tensor]], {}
        )
        output: Dict[str, torch.Tensor] = torch.jit.annotate(
            Dict[str, torch.Tensor], {}
        )
        for i, potential in enumerate(self.models):
            result = potential(data, training, compute_stress, compute_virials, output_index)
            results[str(i)] = result
            for k, v in result.items():
                output[k] = v

        for key in self.out_keys:
            values = []
            for i, potential_key in enumerate(self.potential_keys):
                v_now = results[str(i)][potential_key[key]]
                if 'weight' in potential_key:
                    weight = float(potential_key['weight'])
                    v_now *= weight
                values.append(v_now)
            if values:
                output[key] = self.operation(values)
        return output
