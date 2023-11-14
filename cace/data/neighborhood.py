###########################################################################################
# Neighborhood construction
# modified from MACE
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Optional, Tuple

import ase.neighborlist
import numpy as np


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
    normalized_edge_vectors=True,
    eps: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = 1000. * np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    """
    ‘i’ : first atom index
    ‘j’ : second atom index
    ‘d’ : absolute distance
    ‘D’ : distance vector
    ‘S’ : shift vector (number of cell boundaries crossed by the bond between atom i and j). With the shift vector S, the distances D between atoms can be computed from: D = positions[j]-positions[i]+S.dot(cell)
    """
    sender, receiver, unit_shifts, distance, distance_vector = ase.neighborlist.primitive_neighbor_list(
        quantities="ijSdD",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        self_interaction=True,  # we want edges from atom to itself in different periodic images
        use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]
        distance = distance[keep_edge]
        distance_vector = distance_vector[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    distance = distance[:, np.newaxis]  # [n_edges, 1]   
    if normalized_edge_vectors:
        # Normalize edge vectors
        distance_vector_norm = distance_vector / (distance + eps)
        return edge_index, shifts, unit_shifts, distance, distance_vector_norm
    else:
        # sender = edge_index[0] receiver = edge_index[1]
        return edge_index, shifts, unit_shifts, distance, distance_vector 
