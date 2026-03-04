"""Free-space truncated-kernel Ewald-style potential."""

import math
import warnings
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
        sigma: float = 1.0,
        desired_accuracy: float = 1e-6,
        L: float = 1.8,
        margin: float = 0.9,
        backend: str = "dft",
        feature_key: str = "q",
        output_key: str = "tkm_ewald_potential",
        aggregation_mode: str = "sum",
        compute_field: bool = False,
        remove_self_interaction: bool = True,
    ):
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        if not (0.0 < desired_accuracy < 1.0):
            raise ValueError("desired_accuracy must be in (0, 1).")
        if margin <= 0:
            raise ValueError("margin must be positive.")
        if backend not in ("dft", "nufft"):
            raise ValueError("backend must be either 'dft' or 'nufft'.")

        self.sigma = float(sigma)
        self.sigma_sq_half = 0.5 * self.sigma * self.sigma
        self.desired_accuracy = float(desired_accuracy)
        self.L = float(L)
        self.margin = float(margin)
        self.backend = backend
        self.feature_key = feature_key
        self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        self.compute_field = compute_field
        self.remove_self_interaction = remove_self_interaction

        self.model_outputs = [output_key]
        if compute_field:
            self.model_outputs.append(feature_key + "_field")

        self.delta_k = torch.tensor(torch.pi / 2, dtype=torch.get_default_dtype())
        self.quad_weight = (self.delta_k / (2 * torch.pi)) ** 3
        self.norm_factor = 1.0
        self.last_N = None

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

    def _compute_mode_count(self, scale: torch.Tensor, delta_k: torch.Tensor) -> int:
        k_phys_max = math.sqrt(2.0 * math.log(1.0 / self.desired_accuracy)) / self.sigma
        k_unit_max = k_phys_max * float(scale.item())
        half_modes = max(1, int(math.ceil(k_unit_max / float(delta_k.item()))) + 1)
        return 2 * half_modes

    def compute_potential_tkm_freespace(
        self, r_raw: torch.Tensor, q: torch.Tensor, compute_field: bool = False
    ):
        if self.backend == "dft":
            return self.compute_potential_tkm_freespace_dft(r_raw, q, compute_field)
        if self.backend == "nufft":
            return self.compute_potential_tkm_freespace_nufft(r_raw, q, compute_field)
        raise ValueError(f"Unknown backend: {self.backend}")

    def compute_potential_tkm_freespace_dft(
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

        N = self._compute_mode_count(scale, delta_k)
        self.last_N = N

        idx = torch.arange(N, device=device, dtype=dtype)
        m = idx - (N // 2)
        k1 = delta_k * m

        kx, ky, kz = torch.meshgrid(k1, k1, k1, indexing="ij")
        kvec = torch.stack([kx.reshape(-1), ky.reshape(-1), kz.reshape(-1)], dim=1)  # [K,3]
        k_abs = torch.linalg.norm(kvec, dim=1)
        # Truncated 1/r kernel times Gaussian screening for erf(r/sigma)/r target.
        # k_abs is in unit-box Fourier coordinates, so convert to physical k via /scale.
        ghat = self._truncated_laplace3d_hat(k_abs).to(dtype=dtype)
        k_phys_sq = (k_abs * k_abs) / (scale * scale)
        ghat = ghat * torch.exp(-self.sigma_sq_half * k_phys_sq)

        phase = targets_unit @ kvec.T  # [n,K]
        exp_minus = torch.exp((-1j) * phase).to(dtype=complex_dtype)
        exp_plus = torch.exp((1j) * phase).to(dtype=complex_dtype)

        q_complex = q.to(dtype=complex_dtype)
        rho_hat = exp_minus.T @ q_complex  # [K,nq]
        fk = rho_hat * ghat.to(dtype=complex_dtype).unsqueeze(1)  # [K,nq]

        phi_complex = exp_plus @ fk  # [n,nq]
        phi = quad_weight * torch.real(phi_complex) / scale
        # truncated_laplace3d_hat corresponds to 1/(4*pi*r); CACE electrostatic
        # convention used by EwaldPotential.compute_potential_realspace is 1/(2*pi*r).
        phi = phi * 2.0

        # E = 1/2 sum_i q_i phi_i
        pot = 0.5 * torch.sum(q * phi, dim=0)
        if self.remove_self_interaction:
            twopi = 2.0 * torch.pi
            self_energy = torch.sum(q * q, dim=0) / (self.sigma * (twopi ** 1.5))
            pot = pot - self_energy

        q_field = torch.zeros_like(q, dtype=dtype, device=device)
        if compute_field:
            # For compatibility with existing modules, expose scalar potential-like field per charge channel.
            q_field = phi
            if self.remove_self_interaction:
                twopi = 2.0 * torch.pi
                q_field = q_field - q * (2.0 / (self.sigma * (twopi ** 1.5)))

        return pot * self.norm_factor, q_field * self.norm_factor

    def compute_potential_tkm_freespace_nufft(
        self, r_raw: torch.Tensor, q: torch.Tensor, compute_field: bool = False
    ):
        try:
            from pytorch_finufft import functional as finufft_functional
        except ImportError:
            warnings.warn(
                "pytorch_finufft is unavailable in this environment; falling back to DFT backend.",
                UserWarning,
                stacklevel=2,
            )
            return self.compute_potential_tkm_freespace_dft(r_raw, q, compute_field)

        has_type1 = hasattr(finufft_functional, "finufft_type1")
        has_type2 = hasattr(finufft_functional, "finufft_type2")
        if not (has_type1 and has_type2):
            warnings.warn(
                "pytorch_finufft.functional does not expose finufft_type1/finufft_type2; falling back to DFT backend.",
                UserWarning,
                stacklevel=2,
            )
            return self.compute_potential_tkm_freespace_dft(r_raw, q, compute_field)

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
        N = self._compute_mode_count(scale, delta_k)
        self.last_N = N

        idx = torch.arange(N, device=device, dtype=dtype)
        m = idx - (N // 2)
        k1 = delta_k * m
        kx, ky, kz = torch.meshgrid(k1, k1, k1, indexing="ij")
        k_abs = torch.sqrt(kx * kx + ky * ky + kz * kz)
        # Truncated 1/r kernel times Gaussian screening for erf(r/sigma)/r target.
        # k_abs is in unit-box Fourier coordinates, so convert to physical k via /scale.
        ghat = self._truncated_laplace3d_hat(k_abs).to(dtype=dtype)
        k_phys_sq = (k_abs * k_abs) / (scale * scale)
        ghat = ghat * torch.exp(-self.sigma_sq_half * k_phys_sq)

        # pytorch_finufft points are provided as [D, n_points].
        points = torch.stack(
            [
                delta_k * targets_unit[:, 0],
                delta_k * targets_unit[:, 1],
                delta_k * targets_unit[:, 2],
            ],
            dim=0,
        )

        q_complex = q.to(dtype=complex_dtype)
        n_q = q_complex.shape[1]
        ghat_complex = ghat.to(dtype=complex_dtype)
        phi_channels = []
        for c in range(n_q):
            fk = finufft_functional.finufft_type1(
                points=points,
                values=q_complex[:, c],
                output_shape=(N, N, N),
                eps=self.desired_accuracy,
                isign=-1,
                modeord=0,
            )
            fk = fk * ghat_complex
            phi_c = finufft_functional.finufft_type2(
                points=points,
                targets=fk,
                eps=self.desired_accuracy,
                isign=1,
                modeord=0,
            )
            phi_c = phi_c.reshape(-1)
            if phi_c.shape[0] != r_raw.shape[0]:
                raise RuntimeError(
                    "pytorch_finufft.finufft_type2 returned unexpected output shape."
                )
            phi_channels.append(phi_c)

        phi_complex = torch.stack(phi_channels, dim=1)
        phi = quad_weight * torch.real(phi_complex) / scale
        phi = phi * 2.0
        pot = 0.5 * torch.sum(q * phi, dim=0)
        if self.remove_self_interaction:
            twopi = 2.0 * torch.pi
            self_energy = torch.sum(q * q, dim=0) / (self.sigma * (twopi ** 1.5))
            pot = pot - self_energy

        q_field = torch.zeros_like(q, dtype=dtype, device=device)
        if compute_field:
            q_field = phi
            if self.remove_self_interaction:
                twopi = 2.0 * torch.pi
                q_field = q_field - q * (2.0 / (self.sigma * (twopi ** 1.5)))

        return pot * self.norm_factor, q_field * self.norm_factor
