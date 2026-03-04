# PSWF/SOG Module Testing Notes

Date: 2026-03-04

## Scope

This note covers the merged long-range modules:

- `cace/modules/pswf.py` (`PSWFPotential`)
- `cace/modules/pswf1e3.py` (`PSWFPotential_e3`)
- `cace/modules/pswf_qfield.py` (`PSWFPotential_Qfield`)
- `cace/modules/sog.py` (`SOGPotential`)

## What was verified

1. Syntax/parse checks (`py_compile`) for all added modules:

```bash
python3 -m py_compile \
  cace/modules/pswf.py \
  cace/modules/pswf1e3.py \
  cace/modules/pswf_qfield.py \
  cace/modules/sog.py
```

Result: `PASS`.

2. Export wiring in `cace/modules/__init__.py`:

- `from .pswf import *`
- `from .pswf1e3 import *`
- `from .pswf_qfield import *`
- `from .sog import *`

Result: `PASS`.

3. Import test scaffold added:

- `tests/test_pswf_sog_imports.py`

## Runtime test status

Runtime execution tests were successfully run using a module-loaded Python environment.

Environment setup used:

```bash
module purge
module load python/3.11.11
python3 -m venv .venv-pswf --system-site-packages
. .venv-pswf/bin/activate
python -m pip install ase matscipy
```

Notes:

- `torch` is provided by the loaded module environment (`torch 2.6.0`).
- `ase` and `matscipy` were installed into the local venv for package import compatibility.

## Executed tests and results

1. Import test:

```bash
module purge
module load python/3.11.11
cd ~/codes/cace
. .venv-pswf/bin/activate
python -m pytest -q tests/test_pswf_sog_imports.py
```

Result: `PASS` (`1 passed`).

2. Runtime smoke test (manual script):

- Instantiate each class (`PSWFPotential`, `PSWFPotential_e3`, `PSWFPotential_Qfield`, `SOGPotential`)
- Build a tiny batch dictionary with:
  - `positions` (`[N, 3]`)
  - `cell` (`[1, 3, 3]`)
  - `q` (`[N, 1]`)
  - `batch=None`
- Call `forward` and verify output keys and finite values.

Observed output values from the executed smoke run:

- `pswf_energy`: `[0.00183731]`
- `pswf1e3_energy`: `[0.00126198]`
- `pswfq_energy`: `[0.00183731]`
- `sog_energy`: `[0.00097983]`
- `sogp_energy`: `[0.01103942]`
