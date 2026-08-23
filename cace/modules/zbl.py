"""Differentiable short-range Ziegler--Biersack--Littmark correction.

Source: the ZBL energy expression, screening function ``phi``, and screening
length ``a`` below reproduce the ``ZBLCalculator`` implementation from
hiphive (https://gitlab.com/materials-modeling/hiphive), including its
numerical constants (``prefactor = 14.399645 eV*Angstrom``, the four-term
exponential ``phi``, and ``a = 0.46850 / (Zi**0.23 + Zj**0.23)``). This module
reimplements that formula in PyTorch, in terms of CACE's directed
``edge_index``/``shifts`` graph convention, so it is differentiable through
CACE's ``Forces``/stress pipeline rather than evaluated by a separate ASE
calculator.
"""

from typing import Dict, Optional

import torch
from torch import nn

from .cutoff import SwitchFunction
from .utils import get_edge_vectors_and_lengths
from ..tools.scatter import scatter_sum

__all__ = ["ZBLCorrection"]


class ZBLCorrection(nn.Module):
    """Add a fixed, smoothly switched ZBL pair energy to a CACE graph.

    The ZBL energy formula and its numerical constants are from hiphive's
    ``ZBLCalculator``
    (https://gitlab.com/materials-modeling/hiphive), reimplemented here in
    PyTorch on CACE's directed-edge graph convention so gradients flow
    through CACE's existing ``Forces``/stress machinery.

    The graph contains directed edges, so each edge contribution is weighted
    by one half.  Energies are returned per graph, as required by
    ``FeatureAdd`` and ``Forces``.
    """

    prefactor = 14.399645  # eV Angstrom

    def __init__(
        self,
        switch_on: float = 1.0,
        switch_off: float = 1.5,
        distance_eps: float = 1.0e-4,
        cutoff: Optional[float] = None,
        atomic_numbers_key: str = "atomic_numbers",
        output_key: str = "zbl_energy",
    ):
        super().__init__()
        if switch_on < 0.0 or switch_off <= switch_on:
            raise ValueError("Require 0 <= switch_on < switch_off")
        if distance_eps <= 0.0:
            raise ValueError("distance_eps must be positive")
        if cutoff is not None and cutoff < switch_off:
            raise ValueError("CACE cutoff must be >= switch_off")

        self.switch_on = float(switch_on)
        self.switch_off = float(switch_off)
        self.distance_eps = float(distance_eps)
        self.cutoff = None if cutoff is None else float(cutoff)
        self.atomic_numbers_key = atomic_numbers_key
        self.output_key = output_key
        self.switch = SwitchFunction(switch_on, switch_off)
        self.model_outputs = [output_key]

    def forward(self, data: Dict[str, torch.Tensor], **kwargs):
        positions = data["positions"]
        edge_index = data["edge_index"]
        shifts = data["shifts"]
        atomic_numbers = data[self.atomic_numbers_key]

        batch = data.get("batch")
        if batch is None:
            batch = torch.zeros(
                positions.shape[0], dtype=torch.long, device=positions.device
            )
            num_graphs = 1
        else:
            batch = batch.to(device=positions.device, dtype=torch.long)
            ptr = data.get("ptr")
            num_graphs = int(ptr.numel() - 1) if ptr is not None else (
                int(batch.max().item()) + 1 if batch.numel() else 1
            )

        if edge_index.shape[1] == 0:
            data[self.output_key] = positions.new_zeros((num_graphs,))
            return data

        _, lengths = get_edge_vectors_and_lengths(positions, edge_index, shifts)
        r = lengths.squeeze(-1).clamp_min(self.distance_eps)

        sender = edge_index[0]
        zi = atomic_numbers[sender].to(dtype=r.dtype)
        zj = atomic_numbers[edge_index[1]].to(dtype=r.dtype)
        a = 0.46850 / (zi.pow(0.23) + zj.pow(0.23))
        x = r / a
        phi = (
            0.18175 * torch.exp(-3.19980 * x)
            + 0.50986 * torch.exp(-0.94229 * x)
            + 0.28022 * torch.exp(-0.40290 * x)
            + 0.02817 * torch.exp(-0.20162 * x)
        )
        pair_energy = self.prefactor * zi * zj / r * phi
        pair_energy = 0.5 * pair_energy * self.switch(r)
        graph_index = batch[sender]
        data[self.output_key] = scatter_sum(
            pair_energy, graph_index, dim=0, dim_size=num_graphs
        )
        return data
