import torch

from cace.modules import EwaldPotential


def replicate_box(r, q, box, nx=2, ny=2, nz=2):
    """Replicate the simulation box nx, ny, nz times in each direction."""
    replicated_r = []
    replicated_q = []

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                shift = torch.tensor([ix, iy, iz], dtype=r.dtype, device=r.device) * box
                replicated_r.append(r + shift)
                replicated_q.append(q)

    replicated_r = torch.cat(replicated_r)
    replicated_q = torch.cat(replicated_q)
    new_box = torch.tensor([nx, ny, nz], dtype=r.dtype, device=r.device) * box
    return replicated_r, replicated_q, new_box


def test_ewald_replicated_cell_energy_density_consistency():
    ep = EwaldPotential(
        dl=2,
        sigma=1,
        exponent=1,
        feature_key="q",
        aggregation_mode="sum",
    )

    torch.manual_seed(0)
    r = torch.rand(100, 3, dtype=torch.float64) * 10.0
    q = torch.rand(100, dtype=torch.float64) * 2.0 - 1.0
    box = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)

    replicated_r, replicated_q, new_box = replicate_box(r, q, box, nx=2, ny=2, nz=2)

    ew_1_opt, _ = ep.compute_potential_optimized(r, q.unsqueeze(1), box)
    ew_1_ref, _ = ep.compute_potential(r, q.unsqueeze(1), box)
    ew_2_opt, _ = ep.compute_potential_optimized(replicated_r, replicated_q.unsqueeze(1), new_box)
    ew_2_ref, _ = ep.compute_potential(replicated_r, replicated_q.unsqueeze(1), new_box)

    # Replicating 2x2x2 should scale total energy by ~8 for equivalent density/state.
    assert torch.allclose(ew_1_opt, ew_2_opt / 8.0, rtol=5e-3, atol=5e-3)
    assert torch.allclose(ew_1_ref, ew_2_ref / 8.0, rtol=5e-3, atol=5e-3)
