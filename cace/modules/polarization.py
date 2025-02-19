import torch
import torch.nn as nn
from typing import Dict

__all__ = ['Polarization', 'Dephase', 'FixedCharge']

class Polarization(nn.Module):
    def __init__(self,
                 charge_key: str = 'q',
                 output_key: str = 'polarization',
                 output_index: int = None, # 0, 1, 2 to select only one component
                 phase_key: str = 'phase',
                 remove_mean: bool = True,
                 pbc: bool = False,
                 normalization_factor: float = 1./9.48933,
                 ):
        super().__init__()
        self.charge_key = charge_key
        self.output_key = output_key
        self.output_index = output_index
        self.phase_key = phase_key
        self.model_outputs = [output_key, phase_key]
        self.remove_mean = remove_mean
        self.pbc = pbc
        self.normalization_factor = normalization_factor

    def forward(self, data: Dict[str, torch.Tensor], training=True, output_index=None) -> torch.Tensor:

        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

        box = data['cell'].view(-1, 3, 3)

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
            box_diag = box[i].diagonal(dim1=-2, dim2=-1)
            if box_diag[0] < 1e-6 and box_diag[1] < 1e-6 and box_diag[2] < 1e-6 or self.pbc == False:
                # the box is not periodic, we use the direct sum
                polarization = torch.sum(q_now * r_now, dim=0)
            elif box_diag[0] > 0 and box_diag[1] > 0 and box_diag[2] > 0:
                polarization, phase = self.compute_pol_pbc(r_now, q_now, box_now)
                if self.output_index is not None:
                    phase = phase[:,self.output_index]
                phases.append(phase)
            if self.output_index is not None:
                polarization = polarization[self.output_index]
            results.append(polarization * self.normalization_factor)
        data[self.output_key] = torch.stack(results, dim=0)
        if len(phases) > 0:
            data[self.phase_key] = torch.cat(phases, dim=0)
        else:
            data[self.phase_key] = 0.0
        return data
    
    def compute_pol_pbc(self, r_now, q_now, box_now):
        r_frac = torch.matmul(r_now, torch.linalg.inv(box_now))
        phase = torch.exp(1j * 2.* torch.pi * r_frac)
        S = torch.sum(q_now * phase, dim=0)
        polarization = torch.matmul(box_now.to(S.dtype), 
                                    S.unsqueeze(1)) / (1j * 2.* torch.pi)
        return polarization.reshape(-1), phase

class Dephase(nn.Module):
    def __init__(self,
                 input_key: str = None,
                 phase_key: str = 'phase',
                 output_key: str = 'dephased',
                 input_index: int = None,
                 ):
        super().__init__()
        self.input_key = input_key
        self.input_index = input_index
        self.output_key = output_key
        self.phase_key = phase_key
        self.model_outputs = [output_key]

    def forward(self, data: Dict[str, torch.Tensor], training=None, output_index=None) -> torch.Tensor:
        result = data[self.input_key] * data[self.phase_key].unsqueeze(1).conj()
        data[self.output_key] = result.real
        return data

class FixedCharge(nn.Module):
    def __init__(self,
                 atomic_numbers_key: str = 'atomic_numbers',
                 output_key: str = 'q',
                 charge_dict: Dict[int, float] = None,
                 normalize: bool = True,
                 ):
        super().__init__()
        self.charge_dict = charge_dict
        self.atomic_numbers_key = atomic_numbers_key
        self.output_key = output_key
        self.normalize = normalize
        self.normalization_factor = 9.48933 
    
    def forward(self, data: Dict[str, torch.Tensor], training=None, output_index=None) -> torch.Tensor:
        atomic_numbers = data[self.atomic_numbers_key]
        charge = torch.tensor([self.charge_dict[atomic_number.item()] for atomic_number in atomic_numbers], device=atomic_numbers.device)
        if self.normalize:
            charge = charge * self.normalization_factor # to be consistent with the internal units and the Ewald sum
        data[self.output_key] = charge[:,None]
        return data
