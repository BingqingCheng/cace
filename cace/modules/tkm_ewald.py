"""Free-space truncated-kernel Ewald-style potential."""

from typing import Dict

import torch
import torch.nn as nn

__all__ = ["TKMEwaldPotential"]


class TKMEwaldPotential(nn.Module):
    """
    Truncated-kernel Fourier-space potential for free-space electrostatics.

    This mirrors the core idea used in FBCPoisson.jl:
    1. normalize source/target points to a unit box,
    2. compute Fourier coefficients of charges,
    3. multiply by truncated Laplace kernel in k-space,
    4. evaluate potential back at particle positions.
    """

    def __init__(
        self,
        N: int = 32,
        L: float = 1.8,
        margin: float = 0.9,
        feature_key: str = "q",
        output_key: str = "tkm_ewald_potential",
        aggregation_mode: str = "sum",
        compute_field: bool = False,
        charge_neutral_lambda: float = None,
    ):
        super().__init__()
        if N <= 0:
            raise ValueError("N must be positive.")
        if margin <= 0:
            raise ValueError("margin must be positive.")

        self.N = int(N)
        self.L = float(L)
        self.margin = float(margin)
        self.feature_key = feature_key
        self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        self.compute_field = compute_field
        self.charge_neutral_lambda = charge_neutral_lambda

        self.model_outputs = [output_key]
        if compute_field:
            self.model_outputs.append(feature_key + "_field")

        self.delta_k = torch.tensor(torch.pi / 2, dtype=torch.get_default_dtype())
        self.quad_weight = (self.delta_k / (2 * torch.pi)) ** 3
        self.norm_factor = 1.0

    def forward(self, data: Dict[str, torch.Tensor], **kwargs):
        if data["batch"] is None:
            n_nodes = data["positions"].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data["positions"].device)
        else:
            batch_now = data["batch"]

        r = data["positions"]
        q = data[self.feature_key]
        if q.dim() == 1:
            q = q.unsqueeze(1)

        results = []
        field_results = []
        unique_batches = torch.unique(batch_now)
        for i in unique_batches:
            mask = batch_now == i
            r_now = r[mask]
            q_now = q[mask]
            pot, q_field = self.compute_potential_tkm_freespace(r_now, q_now, self.compute_field)

            if self.charge_neutral_lambda is not None:
                q_mean = torch.mean(q_now)
                pot = pot + self.charge_neutral_lambda * (q_mean**2.0)

            results.append(pot)
            field_results.append(q_field)

        stacked = torch.stack(results, dim=0)
        if self.aggregation_mode == "sum":
            data[self.output_key] = stacked.sum(axis=1)
        else:
            data[self.output_key] = stacked

        if self.compute_field:
            data[self.feature_key + "_field"] = torch.cat(field_results, dim=0)

        return data

    def _truncated_laplace3d_hat(self, k_abs: torch.Tensor) -> torch.Tensor:
        # k=0 -> L^2 / 2 ; otherwise 2*(sin(L*k/2)/k)^2
        out = torch.empty_like(k_abs)
        zero_mask = k_abs == 0
        out[zero_mask] = (self.L**2) / 2.0
        kz = k_abs[~zero_mask]
        out[~zero_mask] = 2.0 * (torch.sin(self.L * kz / 2.0) / kz) ** 2
        return out

    def compute_potential_tkm_freespace(
        self, r_raw: torch.Tensor, q: torch.Tensor, compute_field: bool = False
    ):
        if r_raw.ndim != 2 or r_raw.shape[1] != 3:
            raise ValueError("r_raw must have shape [n_nodes, 3].")
        if q.dim() == 1:
            q = q.unsqueeze(1)
        if q.ndim != 2 or q.shape[0] != r_raw.shape[0]:
            raise ValueError("q must have shape [n_nodes] or [n_nodes, n_q].")

        dtype = r_raw.dtype
        device = r_raw.device
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        mins = torch.min(r_raw, dim=0).values
        maxs = torch.max(r_raw, dim=0).values
        center = 0.5 * (mins + maxs)
        span = torch.max(maxs - mins)
        one = torch.tensor(1.0, dtype=dtype, device=device)
        scale = one if span == 0 else span / self.margin

        targets_unit = (r_raw - center) / scale

        delta_k = self.delta_k.to(device=device, dtype=dtype)
        quad_weight = self.quad_weight.to(device=device, dtype=dtype)

        idx = torch.arange(self.N, device=device, dtype=dtype)
        m = idx - (self.N // 2)
        k1 = delta_k * m

        kx, ky, kz = torch.meshgrid(k1, k1, k1, indexing="ij")
        kvec = torch.stack([kx.reshape(-1), ky.reshape(-1), kz.reshape(-1)], dim=1)  # [K,3]
        k_abs = torch.linalg.norm(kvec, dim=1)
        ghat = self._truncated_laplace3d_hat(k_abs).to(dtype=dtype)

        phase = targets_unit @ kvec.T  # [n,K]
        exp_minus = torch.exp((-1j) * phase).to(dtype=complex_dtype)
        exp_plus = torch.exp((1j) * phase).to(dtype=complex_dtype)

        q_complex = q.to(dtype=complex_dtype)
        rho_hat = exp_minus.T @ q_complex  # [K,nq]
        fk = rho_hat * ghat.to(dtype=complex_dtype).unsqueeze(1)  # [K,nq]

        phi_complex = exp_plus @ fk  # [n,nq]
        phi = quad_weight * torch.real(phi_complex) / scale

        # E = 1/2 sum_i q_i phi_i
        pot = 0.5 * torch.sum(q * phi, dim=0)

        q_field = torch.zeros_like(q, dtype=dtype, device=device)
        if compute_field:
            # For compatibility with existing modules, expose scalar potential-like field per charge channel.
            q_field = phi

        return pot * self.norm_factor, q_field * self.norm_factor
