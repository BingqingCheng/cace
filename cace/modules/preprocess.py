# Description: Preprocess the data for the model

from typing import Dict
import torch
from torch import nn

from .utils import get_symmetric_displacement

__all__ = ["Preprocess"]

class Preprocess(nn.Module): 
    def __init__(self, compute_stress: bool = False, compute_virials: bool = False): 
        super().__init__()
        self.compute_stress = compute_stress
        self.compute_virials = compute_virials

    def forward(self, data: Dict[str, torch.Tensor]): 

        try:
            num_graphs = data["ptr"].numel() - 1
        except:
            num_graphs = 1

        if self.compute_virials or self.compute_stress:
            (
                data["positions"],
                data["shifts"],
                data["displacement"],
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )
        else:
            data["displacement"] = None

        return data


