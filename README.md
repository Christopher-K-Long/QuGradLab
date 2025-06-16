# QuGrad
An extension to the Python package [QuGrad](https://QuGrad.readthedocs.io) that implements common Hilbert space structures, Hamiltonians, and pulse shapes for quantum control.

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
- [QuGrad](https://QuGrad.readthedocs.io)
- [PySTE](https://PySTE.readthedocs.io)
- [TensorFlow](https://www.tensorflow.org)
- [NumPy](https://numpy.org)

#### Additional requirements for testing

- [toml](https://github.com/uiri/toml)
- [PyYAML](https://pyyaml.org/)

## Documentation

Documentation including worked examples can be found at: [https://QuGradLab.readthedocs.io](https://QuGradLab.readthedocs.io)

## Source Code

Source code can be found at: [https://github.com/Christopher-K-Long/QuGradLab](https://github.com/Christopher-K-Long/QuGradLab)

## Version and Changes

The current version is [`0.1.0`](ChangeLog.md#release-010). Please see the [Change Log](ChangeLog.md) for more details. QuGradLab uses [semantic versioning](https://semver.org/).