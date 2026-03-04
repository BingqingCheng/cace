# Repository Guidelines

## Project Structure & Module Organization
- Core package: `cace/`
  - `cace/modules/`: model building blocks (e.g., `ewald.py`, `pswf.py`, `sog.py`, message passing, radial/angular terms).
  - `cace/models/`: model wrappers and composition.
  - `cace/data/`: data conversion and neighborhood graph construction.
  - `cace/tasks/`: training/evaluation utilities.
  - `cace/calculators/`: ASE calculator integration.
  - `cace/tools/`: metrics, scatter ops, and utility helpers.
- Tests: `tests/` (pytest-based unit/smoke checks).
- Examples and benchmarks: `examples/`, `benchmark/`.
- Docs: `README.md`, `docs/`.

## Build, Test, and Development Commands
- Create environment (cluster/module-friendly):
  - `module load python/3.11.11`
  - `python3 -m venv .venv --system-site-packages && . .venv/bin/activate`
- Install local deps:
  - `python -m pip install -e .`
  - If needed: `python -m pip install ase matscipy pytest`
- Run tests:
  - `python -m pytest -q`
  - Targeted: `python -m pytest -q tests/test_pswf_sog_imports.py`
- Quick syntax check:
  - `python -m py_compile cace/modules/*.py`

## Coding Style & Naming Conventions
- Python: PEP 8 style, 4-space indentation, descriptive names.
- Modules/classes: `snake_case.py` files, `PascalCase` class names.
- Keep new modules exportable via `cace/modules/__init__.py`.
- Prefer minimal, focused changes; avoid unrelated refactors in feature PRs.

## Testing Guidelines
- Framework: `pytest`.
- Test files: `tests/test_<feature>.py`.
- Add at least one import/smoke test for new modules and runtime tests when feasible.
- For numerical changes, include finite-value checks and key/output-shape assertions.

## Commit & Pull Request Guidelines
- Commit style in history is short, imperative, and specific (e.g., `Add PSWF/SOG modules...`, `bug fix for reading stress`).
- Use concise subject lines; include scope when helpful (`modules:`, `tasks:`).
- PRs should include:
  - What changed and why.
  - Exact test commands run and results.
  - Any environment assumptions (module loads, optional deps).
  - Backward-compatibility notes for API/output-key changes.
