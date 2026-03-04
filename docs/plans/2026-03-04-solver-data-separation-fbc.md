# Solver/Data Separation for Free-Boundary Workflows Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decouple boundary-condition decisions from data loading so loading never changes system size/cell, enabling clean free-space solvers (including future TKM).

**Architecture:** Keep `AtomicData`/`get_neighborhood` as data-only transforms: read geometry/PBC and build edges without mutating physical box size. Move any “effective box” logic into solver modules (`EwaldPotential` and future TKM). Introduce explicit boundary metadata (`pbc`/`boundary_mode`) so solvers select branches without inferring from modified cell values.

**Tech Stack:** Python, PyTorch, ASE, matscipy, pytest.

---

### Task 1: Lock current behavior with failing tests

**Files:**
- Create: `tests/test_boundary_data_separation.py`

**Step 1: Write failing tests**
- Test A: `get_neighborhood` does not mutate caller `cell` for `pbc=False`.
- Test B: `AtomicData.from_atoms(..., pbc=False, cell=[10,10,10])` preserves nonzero cell diagonals.
- Test C: `EwaldPotential` branching uses explicit boundary metadata, not “cell==0”.

**Step 2: Run tests to verify failure**
Run: `python -m pytest -q tests/test_boundary_data_separation.py`
Expected: FAIL for at least A/B/C.

**Step 3: Commit test scaffold**
```bash
git add tests/test_boundary_data_separation.py
git commit -m "test: add boundary separation regression tests"
```

### Task 2: Stop cell expansion in neighborhood builder

**Files:**
- Modify: `cace/data/neighborhood.py`
- Test: `tests/test_boundary_data_separation.py`

**Step 1: Minimal implementation**
- Remove non-PBC cell expansion (`cell[:, axis] = ...`) from `get_neighborhood`.
- Ensure function operates on local immutable copies where needed.

**Step 2: Run targeted tests**
Run: `python -m pytest -q tests/test_boundary_data_separation.py::test_get_neighborhood_does_not_expand_nonperiodic_cell`
Expected: PASS.

**Step 3: Commit**
```bash
git add cace/data/neighborhood.py tests/test_boundary_data_separation.py
git commit -m "fix(data): keep input cell unchanged for nonperiodic neighborhood build"
```

### Task 3: Add explicit boundary metadata to AtomicData

**Files:**
- Modify: `cace/data/atomic_data.py`
- Test: `tests/test_boundary_data_separation.py`

**Step 1: Minimal implementation**
- Add `pbc` tensor/field into `AtomicData` payload (`from_atoms` should persist ASE `pbc`).
- Preserve physical cell as provided by ASE; do not encode boundary mode by zeroing/enlarging cell.

**Step 2: Run tests**
Run: `python -m pytest -q tests/test_boundary_data_separation.py::test_atomic_data_preserves_nonperiodic_cell`
Expected: PASS.

**Step 3: Commit**
```bash
git add cace/data/atomic_data.py tests/test_boundary_data_separation.py
git commit -m "feat(data): carry explicit pbc metadata in AtomicData"
```

### Task 4: Update Ewald branch selection to use boundary metadata

**Files:**
- Modify: `cace/modules/ewald.py`
- Test: `tests/test_boundary_data_separation.py`

**Step 1: Minimal implementation**
- In `forward()`, select free-space vs periodic branch by `data['pbc']` (or explicit `boundary_mode`) instead of cell-diagonal heuristics.
- Keep current numeric kernels unchanged (only dispatch logic).

**Step 2: Run tests**
Run: `python -m pytest -q tests/test_boundary_data_separation.py::test_ewald_dispatch_uses_boundary_metadata`
Expected: PASS.

**Step 3: Commit**
```bash
git add cace/modules/ewald.py tests/test_boundary_data_separation.py
git commit -m "refactor(ewald): decouple solver dispatch from cell-size heuristics"
```

### Task 5: Validate integration and document behavior

**Files:**
- Modify: `README.md`
- Modify: `docs/pswf-sog-testing.md` (or add `docs/boundary-handling.md`)

**Step 1: Add docs**
- Document invariant: data loader never rescales system size/cell.
- Document solver responsibility: boundary handling and any truncation/box-embedding decisions are solver-side.

**Step 2: Full test run**
Run: `python -m pytest -q`
Expected: PASS (or list known unrelated failures).

**Step 3: Final commit**
```bash
git add README.md docs/pswf-sog-testing.md docs/boundary-handling.md
git commit -m "docs: clarify data-solver boundary separation for free-space workflows"
```
