from typing import Dict, Optional
import torch
from torch import nn

from .utils import get_symmetric_displacement

__all__ = ["Preprocess"]

class Preprocess(nn.Module): 
    def __init__(self): 
        super().__init__()

    def forward(self, data: Dict[str, torch.Tensor], compute_stress: bool = False, compute_virials: bool = False) -> Dict[str, torch.Tensor]: 

        ptr = data.get("ptr")
        if ptr is not None:
            # Ensure that ptr is a Tensor
            assert isinstance(ptr, torch.Tensor)
            num_graphs = ptr.numel() - 1
        else:
            num_graphs = 1

        if compute_virials or compute_stress:
            positions = data["positions"]
            unit_shifts = data["unit_shifts"]
            edge_index = data["edge_index"]
            batch = data["batch"]
            cell = data["cell"]
            
            assert positions is not None, "positions must be provided"
            assert unit_shifts is not None, "unit_shifts must be provided"
            assert edge_index is not None, "edge_index must be provided"
            assert batch is not None, "batch must be provided"
            assert cell is not None, "cell must be provided"
    
            (
                data["positions"],
                data["shifts"],
                data["displacement"],
            ) = get_symmetric_displacement(
                positions=positions,
                unit_shifts=unit_shifts,
                cell=cell,
                edge_index=edge_index,
                num_graphs=num_graphs,
                batch=batch,
            )
        else:
            data["displacement"] = torch.zeros_like(data["positions"])

        return data
