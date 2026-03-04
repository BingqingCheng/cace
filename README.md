# Cartesian Atomic Cluster Expansion for Machine Learning Interatomic Potentials (CACE)

## Summary

The Cartesian Atomic Cluster Expansion (CACE) is a new approach for developing machine learning interatomic potentials. This method utilizes Cartesian coordinates to provide a complete description of atomic environments, maintaining interaction body orders. It integrates low-dimensional embeddings of chemical elements with inter-atomic message passing.

## Requirements

- Python 3.6 or higher
- NumPy
- ASE (Atomic Simulation Environment)
- PyTorch
- matscipy

## Installation

Please refer to the `setup.py` file for installation instructions.

## Usage

Please refer to the `scripts/train.py`.

More example scripts can be found in [https://github.com/BingqingCheng/cacefit].

Long-range MLIP scripts are in [https://github.com/BingqingCheng/cace-lr-fit].

## Added Long-Range Modules (PSWF/SOG)

The following long-range modules are available under `cace.modules`:

- `PSWFPotential` (`cace/modules/pswf.py`)
- `PSWFPotential_e3` (`cace/modules/pswf1e3.py`)
- `PSWFPotential_Qfield` (`cace/modules/pswf_qfield.py`)
- `SOGPotential` (`cace/modules/sog.py`)

They are exported in `cace/modules/__init__.py` and can be imported via:

```python
from cace.modules import PSWFPotential, PSWFPotential_e3, PSWFPotential_Qfield, SOGPotential
```

Testing notes for these modules are documented in `docs/pswf-sog-testing.md`.

## License

This project is licensed under the CC BY-NC 4.0 License - see the LICENSE file for details.

## Citation

```text
@article{cheng2024cartesian,
  title={Cartesian atomic cluster expansion for machine learning interatomic potentials},
  author={Cheng, Bingqing},
  journal={npj Computational Materials},
  volume={10},
  number={1},
  pages={157},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## Contact

For any queries regarding CACE, please contact Bingqing Cheng at tonicbq@gmail.com.
