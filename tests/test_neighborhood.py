import numpy as np
from ase import Atoms

from cace.data.neighborhood import get_neighborhood, get_neighborhood_ASE


def test_get_neighborhood_matches_ase_reference_count():
    atoms = Atoms(
        "OH2",
        positions=[
            [0.000, 0.000, 0.000],
            [0.758, 0.000, 0.504],
            [-0.758, 0.000, 0.504],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )

    cutoff = 5.0
    positions = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    pbc = tuple(atoms.get_pbc())

    edge_index, shifts, unit_shifts = get_neighborhood(
        positions=positions,
        cutoff=cutoff,
        cell=cell.copy(),
        pbc=pbc,
    )
    edge_index_ref, shifts_ref, unit_shifts_ref = get_neighborhood_ASE(
        positions=positions,
        cutoff=cutoff,
        cell=cell.copy(),
        pbc=pbc,
    )

    assert edge_index.shape[0] == 2
    assert shifts.shape[1] == 3
    assert unit_shifts.shape[1] == 3
    # Compare edge counts against ASE-based reference implementation.
    assert edge_index.shape[1] == edge_index_ref.shape[1]
    assert shifts.shape == shifts_ref.shape
    assert unit_shifts.shape == unit_shifts_ref.shape
