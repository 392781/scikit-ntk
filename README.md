## Neural Tangent Kernel for `scikit-learn` Gaussian Processes

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/392781/scikit-ntk/Lint,%20Build,%20Install,%20Test?label=Lint%2C%20Build%2C%20Install%2C%20Test&style=flat-square) ![PyPI](https://img.shields.io/pypi/v/scikit-ntk?style=flat-square) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scikit-ntk?style=flat-square)

![PyPI - Downloads](https://img.shields.io/pypi/dm/scikit-ntk?style=flat-square)

**scikit-ntk** is implementation of the neural tangent kernel (NTK) for the `scikit-learn` machine learning library as part of "An Empirical Analysis of the Laplace and Neural Tangent Kernels" master's thesis (found at [http://hdl.handle.net/20.500.12680/d504rr81v](http://hdl.handle.net/20.500.12680/d504rr81v) and [https://arxiv.org/abs/2208.03761](https://arxiv.org/abs/2208.03761)).  This library is meant to directly integrate with [`sklearn.gaussian_process`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process) module.  This implementation of the NTK can be used in combination with other kernels to train and predict with Gaussian process regressors and classifiers. 

## Installation

### Dependencies

scikit-ntk requires:
* Python (>=3.7)
* scikit-learn (>=1.0.1)


### User installation
In terminal using `pip` run:

```bash
pip install scikit-ntk
```

### Usage
Usage is described in [`examples/usage.py`](https://github.com/392781/scikit-ntk/blob/master/example/usage.py); however, to get started simply import the `NeuralTangentKernel` class:

```py
from skntk import NeuralTangentKernel as NTK

kernel_ntk = NTK(D=3, bias=0.01, bias_bounds=(1e-6, 1e6))
```
Once declared, usage is the same as other `scikit-learn` kernels.

## Building
Python Poetry (>=1.2) is required if you wish to build `scikit-ntk` from source.  In order to build follow these steps:

1. Clone the repository
```bash
git clone git@github.com:392781/scikit-ntk.git
```
2. Enable a Poetry virtual environment
```bash
poetry shell
```
3. Build and install
```bash
poetry build
poetry install --with dev
```

## Citation

If you use scikit-ntk in your scientific work, please use the following citation alongside the scikit-learn citations found at [https://scikit-learn.org/stable/about.html#citing-scikit-learn](https://scikit-learn.org/stable/about.html#citing-scikit-learn):

```
@mastersthesis{lencevicius2022laplacentk,
  author  = "Ronaldas Paulius Lencevicius",
  title   = "An Empirical Analysis of the Laplace and Neural Tangent Kernels",
  school  = "California State Polytechnic University, Pomona",
  year    = "2022",
  month   = "August",
  note    = {\url{http://hdl.handle.net/20.500.12680/d504rr81v}}
}
```
