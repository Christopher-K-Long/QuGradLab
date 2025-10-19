# QuGradLab
An extension to the Python package [QuGrad](https://QuGrad.readthedocs.io) ([doi:10.5281/zenodo.17116721](https://doi.org/10.5281/zenodo.17116721)) that implements common Hilbert space structures, Hamiltonians, and pulse shapes for quantum control.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17116725.svg)](https://doi.org/10.5281/zenodo.17116725)

[![Unit Tests](https://github.com/Christopher-K-Long/QuGradLab/actions/workflows/test-python-package.yml/badge.svg)](https://github.com/Christopher-K-Long/QuGradLab/actions/workflows/test-python-package.yml)

## Installation

The python package can be installed with pip as follows:
```bash
pip install qugradlab
```

If on Linux and using a conda environment you may encounter an error
```
version `GLIBCXX_...' not found
```
to fix this you also need to execute:
```bash
conda install -c conda-forge libstdcxx-ng
```

### Requirements

Requires:
- [QuGrad](https://QuGrad.readthedocs.io) (== 1.*) ([doi:10.5281/zenodo.17116721](https://doi.org/10.5281/zenodo.17116721))
- [PySTE](https://PySTE.readthedocs.io) (== 1.*) ([doi:10.5281/zenodo.17116431](https://doi.org/10.5281/zenodo.17116431))
- [TensorFlow](https://www.tensorflow.org) (== 2.*)
- [NumPy](https://numpy.org) (>= 1.21, < 3)
- [SciPy](https://scipy.org/) (== 1.*)

#### Additional requirements for testing

- [toml](https://github.com/uiri/toml)
- [PyYAML](https://pyyaml.org/)

## Documentation

Documentation including worked examples can be found at: [https://QuGradLab.readthedocs.io](https://QuGradLab.readthedocs.io)

## Source Code

Source code can be found at: [https://github.com/Christopher-K-Long/QuGradLab](https://github.com/Christopher-K-Long/QuGradLab)

## Version and Changes

The current version is [`0.1.1`](ChangeLog.md#release-011). Please see the [Change Log](ChangeLog.md) for more details. QuGradLab uses [semantic versioning](https://semver.org/).