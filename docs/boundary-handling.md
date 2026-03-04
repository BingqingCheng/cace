# Boundary Handling in CACE

This repository now follows a strict separation between **data loading** and **solver boundary logic**.

## Data-side behavior

- `cace.data.neighborhood.get_neighborhood` no longer rescales/expands the user-provided cell for non-periodic axes.
- `AtomicData.from_atoms` preserves the input ASE cell and stores explicit `pbc` metadata in the graph payload.

In short: data loading should represent the physical system as given.

## Solver-side behavior

- Boundary-mode branch selection is solver responsibility.
- `EwaldPotential.forward` now prefers explicit `data["pbc"]` metadata when available:
  - all `False` -> free-space real-space branch (`compute_potential_realspace`, currently exponent=1)
  - any `True` -> reciprocal/triclinic branch (`compute_potential_triclinic`)
- Legacy cell-diagonal dispatch remains as fallback for payloads without `pbc`.

## Regression tests

See `tests/test_boundary_data_separation.py`:

1. neighborhood builder does not mutate non-periodic input cell
2. `AtomicData` preserves non-periodic input cell
3. Ewald dispatch uses boundary metadata instead of cell-size heuristics
