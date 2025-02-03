import torch
import torch.nn as nn
from typing import Dict

__all__ = ['Polarization', 'Dephase', 'FixedCharge']

class Polarization(nn.Module):
    def __init__(self,
                 charge_key: str = 'q',
                 output_key: str = 'polarization',
                 phase_key: str = 'phase',
                 remove_mean: bool = True,
                 pbc: bool = False,
                 ):
        super().__init__()
        self.charge_key = charge_key
        self.output_key = output_key
        self.phase_key = phase_key
        self.model_outputs = [output_key, phase_key]
        self.remove_mean = remove_mean
        self.pbc = pbc

    def forward(self, data: Dict[str, torch.Tensor], training=True, output_index=None) -> torch.Tensor:

        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

        box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1)

        r = data['positions']
        q = data[self.charge_key]

        if q.dim() == 1:
            q = q.unsqueeze(1)
        if self.remove_mean:
            q = q - torch.mean(q, dim=0, keepdim=True)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'

        unique_batches = torch.unique(batch_now)  # Get unique batch indices

        results = []
        phases = [] 
        for i in unique_batches:
            mask = batch_now == i  # Create a mask for the i-th configuration
            r_now, q_now, box_now = r[mask], q[mask], box[i]
            if box_now[0] < 1e-6 and box_now[1] < 1e-6 and box_now[2] < 1e-6 or self.pbc == False:
                # the box is not periodic, we use the direct sum
                polarization = torch.sum(q_now * r_now, dim=0)
            elif box_now[0] > 0 and box_now[1] > 0 and box_now[2] > 0:
                factor = box_now / (1j * 2.* torch.pi)
                phase = torch.exp(1j * 2.* torch.pi * r_now / box_now)
                polarization = torch.sum(q_now * phase, dim=(0)) * factor
                phases.append(phase)
            results.append(polarization)
        data[self.output_key] = torch.stack(results, dim=0)
        if len(phases) > 0:
            data[self.phase_key] = torch.stack(phases, dim=0)
        else:
            data[self.phase_key] = 0.0
        return data

class Dephase(nn.Module):
    def __init__(self,
                 input_key: str = None,
                 phase_key: str = 'phase',
                 output_key: str = 'dephased',
                 ):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.phase_key = phase_key
        self.model_outputs = [output_key]

    def forward(self, data: Dict[str, torch.Tensor], training=None, output_index=None) -> torch.Tensor:
        result = data[self.input_key] * data[self.phase_key].unsqueeze(-2).conj()
        data[self.output_key] = result.real
        return data

class FixedCharge(nn.Module):
    def __init__(self,
                 atomic_numbers_key: str = 'atomic_numbers',
                 output_key: str = 'q',
                 charge_dict: Dict[int, float] = None,
                 normalize: bool = True,
                 trainable: bool = False,
                 ):
        super().__init__()
        self.atomic_numbers_key = atomic_numbers_key
        self.output_key = output_key
        self.normalize = normalize
        self.normalization_factor = 9.48933 
        self.elements = [ key for key in charge_dict.keys()]
        element_charges = torch.tensor([charge_dict[key] for key in charge_dict.keys()], dtype=torch.float32)
        if trainable:
            self.element_charges = nn.Parameter(element_charges)
        else:
            self.register_buffer("element_charges", element_charges)
    
    def get_charge_dict(self):
        return {element: self.element_charges[i] for i, element in enumerate(self.elements)}

    def forward(self, data: Dict[str, torch.Tensor], training=None, output_index=None) -> torch.Tensor:
        atomic_numbers = data[self.atomic_numbers_key]
        charge = torch.zeros(atomic_numbers.shape[0], device=atomic_numbers.device)
        for i, element in enumerate(self.elements):
            mask = atomic_numbers == element
            charge[mask] = self.element_charges[i]
        if self.normalize:
            charge = charge * self.normalization_factor # to be consistent with the internal units and the Ewald sum
        data['q'] = charge
        return data
