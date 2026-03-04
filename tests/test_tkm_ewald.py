import torch

from cace.modules import TKMEwaldPotential


def test_tkm_ewald_forward_outputs_finite_energy():
    m = TKMEwaldPotential(N=16, L=1.8, margin=0.9, feature_key="q", output_key="tkm_e")
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
    m = TKMEwaldPotential(N=16, L=1.8, margin=0.9, feature_key="q", output_key="tkm_e")
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
    m = TKMEwaldPotential(N=12, L=1.8, margin=0.9, feature_key="q", output_key="tkm_e")
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
