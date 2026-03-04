import numpy as np
import torch
from ase import Atoms

from cace.data.atomic_data import AtomicData
from cace.data.neighborhood import get_neighborhood
from cace.modules.ewald import EwaldPotential


def test_get_neighborhood_does_not_expand_nonperiodic_cell():
    positions = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=float)
    cell = np.diag([10.0, 10.0, 10.0]).astype(float)
    cell_before = cell.copy()

    get_neighborhood(
        positions=positions,
        cutoff=4.0,
        pbc=(False, False, False),
        cell=cell,
    )

    # Data loading should not mutate physical system size/cell in-place.
    np.testing.assert_allclose(cell, cell_before)


def test_atomic_data_preserves_nonperiodic_cell():
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )

    data = AtomicData.from_atoms(atoms=atoms, cutoff=4.0)
    diag = data["cell"].diag().detach().cpu().numpy()

    np.testing.assert_allclose(diag, np.array([10.0, 10.0, 10.0]))


def test_ewald_dispatch_uses_boundary_metadata():
    m = EwaldPotential(exponent=1, feature_key="q", output_key="ew", compute_field=False)
    calls = {"realspace": 0, "triclinic": 0}

    orig_real = m.compute_potential_realspace
    orig_tri = m.compute_potential_triclinic

    def real_wrap(*args, **kwargs):
        calls["realspace"] += 1
        return orig_real(*args, **kwargs)

    def tri_wrap(*args, **kwargs):
        calls["triclinic"] += 1
        return orig_tri(*args, **kwargs)

    m.compute_potential_realspace = real_wrap
    m.compute_potential_triclinic = tri_wrap

    data = {
        "batch": None,
        "positions": torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=torch.float32),
        "cell": torch.eye(3, dtype=torch.float32).unsqueeze(0) * 10.0,
        "q": torch.tensor([[0.5], [-0.5]], dtype=torch.float32),
        # boundary metadata should force free-space branch regardless of positive cell size
        "pbc": torch.tensor([False, False, False]),
    }

    m(data)
    assert calls["realspace"] == 1
    assert calls["triclinic"] == 0
