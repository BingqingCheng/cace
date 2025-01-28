import torch
import torch.nn as nn
from typing import Dict

__all__ = ['Polarization']

class Polarization(nn.Module):
    def __init__(self,
                 charge_key: str = 'q',
                 output_key: str = 'polarization',
                 remove_mean: bool = True,
                 ):
        super().__init__()
        self.charge_key = charge_key
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.remove_mean = remove_mean

    def forward(self, data: Dict[str, torch.Tensor], training=None, output_index=None) -> torch.Tensor:

        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

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
        for i in unique_batches:
            mask = batch_now == i  # Create a mask for the i-th configuration
            r_now, q_now = r[mask], q[mask]
            polarization = torch.sum(q_now * r_now, dim=0)

            results.append(polarization)

        data[self.output_key] = torch.stack(results, dim=0)
        return data
