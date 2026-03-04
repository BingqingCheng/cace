import pytest
import torch
import types

from cace.modules import EwaldPotential, TKMEwaldPotential


def test_tkm_ewald_forward_outputs_finite_energy():
    m = TKMEwaldPotential(
        sigma=1.0, desired_accuracy=1e-6, L=1.8, margin=0.9, feature_key="q", output_key="tkm_e"
    )
    data = {
        "batch": None,
        "positions": torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, -0.2]], dtype=torch.float32),
        "q": torch.tensor([[0.5], [-0.5]], dtype=torch.float32),
    }

    out = m(data)
    assert "tkm_e" in out
    assert out["tkm_e"].shape == (1,)
    assert torch.isfinite(out["tkm_e"]).all()


def test_tkm_ewald_translation_invariance():
    m = TKMEwaldPotential(
        sigma=1.0, desired_accuracy=1e-6, L=1.8, margin=0.9, feature_key="q", output_key="tkm_e"
    )
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.1, -0.2], [-0.4, 0.7, 0.3]],
        dtype=torch.float64,
    )
    q = torch.tensor([[0.4], [-0.3], [0.1]], dtype=torch.float64)
    shift = torch.tensor([3.2, -1.4, 2.7], dtype=torch.float64)

    out_a = m({"batch": None, "positions": pos, "q": q})["tkm_e"]
    out_b = m({"batch": None, "positions": pos + shift, "q": q})["tkm_e"]
    assert torch.allclose(out_a, out_b, rtol=1e-8, atol=1e-8)


def test_tkm_ewald_batch_support():
    m = TKMEwaldPotential(
        sigma=1.0, desired_accuracy=1e-6, L=1.8, margin=0.9, feature_key="q", output_key="tkm_e"
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.1, 0.0],
            [2.0, 2.0, 2.0],
            [2.3, 2.1, 2.2],
        ],
        dtype=torch.float32,
    )
    q = torch.tensor([[0.4], [-0.4], [0.2], [-0.2]], dtype=torch.float32)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    out = m({"batch": batch, "positions": positions, "q": q})["tkm_e"]
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_tkm_ewald_matches_ewald_realspace_energy():
    # Match sigma so both models target the same screened erf(r/sigma)/r interaction.
    ewald = EwaldPotential(
        sigma=0.3,
        exponent=1,
        feature_key="q",
        output_key="ew",
        compute_field=False,
        remove_self_interaction=True,
    )
    tkm = TKMEwaldPotential(
        sigma=0.3, desired_accuracy=1e-6, L=1.8, margin=0.9, feature_key="q", output_key="tkm_e"
    )

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.1, -0.2],
            [-0.4, 0.7, 0.3],
            [0.2, -0.6, 0.5],
        ],
        dtype=torch.float64,
    )
    q = torch.tensor([[0.4], [-0.3], [0.1], [-0.2]], dtype=torch.float64)

    e_real, _ = ewald.compute_potential_realspace(positions, q, compute_field=False)
    e_tkm = tkm({"batch": None, "positions": positions, "q": q})["tkm_e"]

    # The free-space TKM should agree with direct realspace energy on this small system.
    assert torch.allclose(e_tkm, e_real, rtol=5e-2, atol=5e-2)


def test_tkm_ewald_mode_count_increases_with_tighter_accuracy():
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, -0.2]], dtype=torch.float64)
    q = torch.tensor([[0.5], [-0.5]], dtype=torch.float64)

    loose = TKMEwaldPotential(
        sigma=1.0, desired_accuracy=1e-4, L=1.8, margin=0.9, feature_key="q", output_key="e_loose"
    )
    tight = TKMEwaldPotential(
        sigma=1.0, desired_accuracy=1e-8, L=1.8, margin=0.9, feature_key="q", output_key="e_tight"
    )

    loose({"batch": None, "positions": positions, "q": q})
    tight({"batch": None, "positions": positions, "q": q})

    assert loose.last_N is not None and tight.last_N is not None
    assert tight.last_N >= loose.last_N


def test_tkm_ewald_backend_dft_explicit():
    m = TKMEwaldPotential(
        sigma=1.0,
        desired_accuracy=1e-6,
        backend="dft",
        feature_key="q",
        output_key="tkm_e",
    )
    out = m(
        {
            "batch": None,
            "positions": torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, -0.2]], dtype=torch.float32),
            "q": torch.tensor([[0.5], [-0.5]], dtype=torch.float32),
        }
    )["tkm_e"]
    assert torch.isfinite(out).all()


def test_tkm_ewald_backend_nufft_uses_pytorch_finufft_when_available(monkeypatch):
    call_counts = {"type1": 0, "type2": 0}

    def fake_type1(points, values, output_shape, **kwargs):
        call_counts["type1"] += 1
        assert points.shape[0] == 3
        assert values.ndim == 1
        return torch.zeros(output_shape, dtype=values.dtype, device=values.device)

    def fake_type2(points, targets, **kwargs):
        call_counts["type2"] += 1
        n = points.shape[1]
        return torch.zeros(n, dtype=targets.dtype, device=targets.device)

    fake_functional = types.SimpleNamespace(finufft_type1=fake_type1, finufft_type2=fake_type2)
    fake_module = types.SimpleNamespace(functional=fake_functional)
    monkeypatch.setitem(__import__("sys").modules, "pytorch_finufft", fake_module)

    m = TKMEwaldPotential(
        sigma=1.0, desired_accuracy=1e-6, backend="nufft", feature_key="q", output_key="e_nufft"
    )

    def dft_should_not_be_called(*args, **kwargs):
        raise RuntimeError("DFT fallback should not be used when pytorch_finufft is present")

    monkeypatch.setattr(m, "compute_potential_tkm_freespace_dft", dft_should_not_be_called)

    out = m(
        {
            "batch": None,
            "positions": torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, -0.2]], dtype=torch.float64),
            "q": torch.tensor([[0.5], [-0.5]], dtype=torch.float64),
        }
    )["e_nufft"]

    assert torch.isfinite(out).all()
    assert call_counts["type1"] == 1
    assert call_counts["type2"] == 1


@pytest.mark.parametrize("sigma", [0.3, 1.0])
def test_tkm_ewald_cross_compare_100_atoms_realspace_dft_nufft(sigma):
    pytest.importorskip("pytorch_finufft")

    torch.manual_seed(1234)
    n_atoms = 100
    positions = torch.rand(n_atoms, 3, dtype=torch.float64) * 5.0
    q = (torch.rand(n_atoms, 1, dtype=torch.float64) * 2.0 - 1.0)
    q = q - q.mean(dim=0, keepdim=True)

    ewald = EwaldPotential(
        sigma=sigma,
        exponent=1,
        feature_key="q",
        output_key="ew",
        compute_field=False,
        remove_self_interaction=True,
    )
    tkm_dft = TKMEwaldPotential(
        sigma=sigma, desired_accuracy=1e-4, backend="dft", feature_key="q", output_key="e_dft"
    )
    tkm_nufft = TKMEwaldPotential(
        sigma=sigma, desired_accuracy=1e-4, backend="nufft", feature_key="q", output_key="e_nufft"
    )

    e_ewald, _ = ewald.compute_potential_realspace(positions, q, compute_field=False)
    e_dft = tkm_dft({"batch": None, "positions": positions, "q": q})["e_dft"]
    e_nufft = tkm_nufft({"batch": None, "positions": positions, "q": q})["e_nufft"]

    assert torch.isfinite(e_ewald).all()
    assert torch.isfinite(e_dft).all()
    assert torch.isfinite(e_nufft).all()
    assert torch.allclose(e_dft, e_nufft, rtol=5e-3, atol=5e-3)
    assert torch.allclose(e_dft, e_ewald, rtol=5e-3, atol=5e-3)
    assert torch.allclose(e_nufft, e_ewald, rtol=5e-3, atol=5e-3)


def test_tkm_ewald_remove_self_interaction_matches_realspace_single_charge():
    pytest.importorskip("pytorch_finufft")

    sigma = 0.3
    positions = torch.tensor([[1.2, -0.3, 0.7]], dtype=torch.float64)
    q = torch.tensor([[1.0]], dtype=torch.float64)

    ewald = EwaldPotential(
        sigma=sigma,
        exponent=1,
        feature_key="q",
        output_key="ew",
        compute_field=False,
        remove_self_interaction=True,
    )
    tkm_drop = TKMEwaldPotential(
        sigma=sigma,
        desired_accuracy=1e-6,
        backend="dft",
        feature_key="q",
        output_key="e_drop",
        remove_self_interaction=True,
    )
    tkm_keep = TKMEwaldPotential(
        sigma=sigma,
        desired_accuracy=1e-6,
        backend="dft",
        feature_key="q",
        output_key="e_keep",
        remove_self_interaction=False,
    )

    e_ewald, _ = ewald.compute_potential_realspace(positions, q, compute_field=False)
    e_drop = tkm_drop({"batch": None, "positions": positions, "q": q})["e_drop"]
    e_keep = tkm_keep({"batch": None, "positions": positions, "q": q})["e_keep"]

    twopi = 2.0 * torch.pi
    expected_self = torch.sum(q * q, dim=0) / (sigma * (twopi ** 1.5))
    assert torch.allclose(e_ewald, torch.zeros_like(e_ewald), atol=1e-12, rtol=0.0)
    assert torch.allclose(e_drop, e_ewald, atol=5e-8, rtol=1e-8)
    assert torch.allclose(e_keep - e_drop, expected_self, atol=5e-8, rtol=1e-8)
