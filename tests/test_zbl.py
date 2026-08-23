"""Tests for cace.modules.zbl.ZBLCorrection.

Sources:

* The ZBL energy formula and reference constants (``hiphive_energy`` below)
  reproduce hiphive's ``ZBLCalculator``
  (https://gitlab.com/materials-modeling/hiphive), which is also the source
  ZBLCorrection itself was ported from (see cace/modules/zbl.py).
* The tests independently re-derive reference values from the closed-form
  ZBL expression rather than reading them from the implementation.
"""

import pytest
import torch

from cace.modules import FeatureAdd, Forces, ZBLCorrection


def pair_data(distance=0.8, dtype=torch.float64, directed=True):
    positions = torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=dtype)
    positions.requires_grad_(True)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    if not directed:
        edge_index = edge_index[:, :1]
    return {
        "positions": positions,
        "edge_index": edge_index,
        "shifts": torch.zeros((edge_index.shape[1], 3), dtype=dtype),
        "atomic_numbers": torch.tensor([28, 17], dtype=torch.long),
        "batch": torch.zeros(2, dtype=torch.long),
        "ptr": torch.tensor([0, 2], dtype=torch.long),
    }


def analytic_energy(zi, zj, r):
    a = 0.46850 / (zi**0.23 + zj**0.23)
    x = r / a
    phi = (
        0.18175 * torch.exp(-3.19980 * x)
        + 0.50986 * torch.exp(-0.94229 * x)
        + 0.28022 * torch.exp(-0.40290 * x)
        + 0.02817 * torch.exp(-0.20162 * x)
    )
    return 14.399645 * zi * zj / r * phi


def test_analytic_pair_and_directed_edge_accounting():
    module = ZBLCorrection(switch_on=1.0, switch_off=1.5)
    data = pair_data()
    result = module(data)["zbl_energy"][0]
    dtype = data["positions"].dtype
    expected = analytic_energy(
        torch.tensor(28.0, dtype=dtype),
        torch.tensor(17.0, dtype=dtype),
        torch.tensor(0.8, dtype=dtype),
    )
    assert torch.allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_switch_is_zero_above_cutoff():
    module = ZBLCorrection(switch_on=1.0, switch_off=1.5)
    assert module(pair_data(1.6))["zbl_energy"].item() == 0.0


def test_force_gradient_includes_zbl():
    module = ZBLCorrection(switch_on=1.0, switch_off=1.5)
    data = pair_data(0.8)
    energy = module(data)["zbl_energy"]
    force = -torch.autograd.grad(energy.sum(), data["positions"])[0]
    expected = torch.autograd.functional.jacobian(
        lambda p: module({**data, "positions": p})["zbl_energy"].sum(),
        data["positions"],
    )
    assert torch.allclose(force, -expected, rtol=1e-8, atol=1e-8)


def test_empty_edges_and_float32_clamp_are_finite():
    module = ZBLCorrection()
    data = pair_data(1.0e-8, dtype=torch.float32)
    energy = module(data)["zbl_energy"]
    assert torch.isfinite(energy).all()
    assert torch.isfinite(torch.autograd.grad(energy.sum(), data["positions"])[0]).all()

    data["edge_index"] = torch.empty((2, 0), dtype=torch.long)
    data["shifts"] = torch.empty((0, 3), dtype=torch.float32)
    assert module(data)["zbl_energy"].shape == (1,)
    assert module(data)["zbl_energy"].item() == 0.0


def test_total_energy_force_ordering():
    data = pair_data(0.8)
    data["learned"] = torch.zeros(1, dtype=torch.float64)
    zbl = ZBLCorrection()(data)["zbl_energy"]
    data["zbl_energy"] = zbl
    total = FeatureAdd(["learned", "zbl_energy"], "energy")(data)
    output = Forces(
        energy_key="energy", forces_key="forces", calc_stress=False
    )(total, training=False)
    assert torch.allclose(output["energy"], zbl)
    assert torch.isfinite(output["forces"]).all()


def test_cutoff_validation():
    with pytest.raises(ValueError):
        ZBLCorrection(switch_on=1.0, switch_off=2.0, cutoff=1.5)


def test_periodic_self_loop_pair():
    """An atom interacting with itself through nonzero periodic images
    (sender == receiver, shift != 0) must be treated as an ordinary weighted
    pair. CACE's get_neighborhood only drops a self-loop when unit_shifts is
    exactly zero (neighborhood.py:16-58), so self-loops through a periodic
    image are real inputs the module must not special-case or NaN on."""
    module = ZBLCorrection(switch_on=2.5, switch_off=3.0)
    dtype = torch.float64
    positions = torch.zeros((1, 3), dtype=dtype, requires_grad=True)
    # Two distinct self-loop edges: the atom's own periodic images at +2 and
    # -2 Angstrom along x, exactly as a real neighbor list would return.
    # Both distances (r=2.0) are below switch_on=2.5, so the switch factor
    # is exactly 1 and the analytic hiphive comparison is exact.
    edge_index = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    shifts = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], dtype=dtype)
    data = {
        "positions": positions,
        "edge_index": edge_index,
        "shifts": shifts,
        "atomic_numbers": torch.tensor([28], dtype=torch.long),
        "batch": torch.zeros(1, dtype=torch.long),
        "ptr": torch.tensor([0, 1], dtype=torch.long),
    }
    result = module(data)["zbl_energy"][0]
    expected = analytic_energy(
        torch.tensor(28.0, dtype=dtype),
        torch.tensor(28.0, dtype=dtype),
        torch.tensor(2.0, dtype=dtype),
    )
    # Both edges have equal length and each carries weight 0.5, so they sum
    # to exactly one hiphive pair value -- confirms uniform per-edge
    # weighting with no self-loop special-casing.
    assert torch.isfinite(result)
    assert torch.allclose(result, expected, rtol=1e-12, atol=1e-12)
    grad = torch.autograd.grad(result, positions, retain_graph=True)[0]
    assert torch.isfinite(grad).all()


def test_batching_matches_independent_evaluation():
    """Two structures evaluated in one batch must each get their own
    zbl_energy entry, equal to evaluating each structure independently."""
    module = ZBLCorrection(switch_on=1.0, switch_off=1.5)
    dtype = torch.float64

    data_a = pair_data(distance=0.8, dtype=dtype)
    data_b = pair_data(distance=1.1, dtype=dtype)
    data_b["atomic_numbers"] = torch.tensor([11, 17], dtype=torch.long)

    energy_a = module(data_a)["zbl_energy"]
    energy_b = module(data_b)["zbl_energy"]

    batched_positions = torch.cat(
        [data_a["positions"].detach(), data_b["positions"].detach()], dim=0
    )
    batched_positions.requires_grad_(True)
    batched_edge_index = torch.cat(
        [data_a["edge_index"], data_b["edge_index"] + 2], dim=1
    )
    batched_shifts = torch.cat([data_a["shifts"], data_b["shifts"]], dim=0)
    batched_atomic_numbers = torch.cat(
        [data_a["atomic_numbers"], data_b["atomic_numbers"]], dim=0
    )
    batched_data = {
        "positions": batched_positions,
        "edge_index": batched_edge_index,
        "shifts": batched_shifts,
        "atomic_numbers": batched_atomic_numbers,
        "batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "ptr": torch.tensor([0, 2, 4], dtype=torch.long),
    }
    batched_energy = module(batched_data)["zbl_energy"]

    assert batched_energy.shape == (2,)
    assert torch.allclose(batched_energy[0], energy_a[0], rtol=1e-12, atol=1e-12)
    assert torch.allclose(batched_energy[1], energy_b[0], rtol=1e-12, atol=1e-12)


def _apply_strain(positions, unit_shifts, cell, edge_index, batch, displacement):
    """Reproduce cace.modules.utils.get_symmetric_displacement's position/
    shift transformation (utils.py:110-145) without allocating a fresh
    zero displacement leaf, so a caller-chosen displacement value can be
    probed both for autograd and for finite differences."""
    symmetric = 0.5 * (displacement + displacement.transpose(-1, -2))
    new_positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric[batch]
    )
    new_cell = cell + torch.matmul(cell, symmetric)
    sender = edge_index[0]
    new_shifts = torch.einsum(
        "be,bec->bc", unit_shifts, new_cell[batch[sender]]
    )
    return new_positions, new_shifts


def test_stress_virial_matches_finite_difference():
    """ZBL must contribute to the virial computed via CACE's strain-trick
    displacement tensor (get_symmetric_displacement, utils.py:110-145),
    which Forces/compute_forces_virials differentiates against -- not
    against cell or positions directly."""
    dtype = torch.float64
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=dtype)
    cell = (torch.eye(3, dtype=dtype) * 5.0).unsqueeze(0)  # [1, 3, 3]
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    unit_shifts = torch.zeros((2, 3), dtype=dtype)
    batch = torch.zeros(2, dtype=torch.long)
    atomic_numbers = torch.tensor([28, 17], dtype=torch.long)
    module = ZBLCorrection(switch_on=1.0, switch_off=1.5)

    def zbl_energy_for_displacement(displacement):
        pos, shifts = _apply_strain(
            positions, unit_shifts, cell, edge_index, batch, displacement
        )
        data = {
            "positions": pos,
            "edge_index": edge_index,
            "shifts": shifts,
            "atomic_numbers": atomic_numbers,
            "batch": batch,
            "ptr": torch.tensor([0, 2], dtype=torch.long),
        }
        return module(data)["zbl_energy"].sum()

    displacement = torch.zeros(1, 3, 3, dtype=dtype, requires_grad=True)
    energy = zbl_energy_for_displacement(displacement)
    virial = torch.autograd.grad(energy, displacement)[0][0]
    assert torch.isfinite(virial).all()

    h = 1.0e-6
    fd = torch.zeros(3, 3, dtype=dtype)
    for i in range(3):
        for j in range(3):
            dp = torch.zeros(1, 3, 3, dtype=dtype)
            dp[0, i, j] = h
            dm = torch.zeros(1, 3, 3, dtype=dtype)
            dm[0, i, j] = -h
            e_plus = zbl_energy_for_displacement(dp).item()
            e_minus = zbl_energy_for_displacement(dm).item()
            fd[i, j] = (e_plus - e_minus) / (2 * h)

    assert torch.allclose(virial, fd, rtol=1e-4, atol=1e-8)
