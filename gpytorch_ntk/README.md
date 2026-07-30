# GPyTorch Neural Tangent Kernel

This directory contains a GPyTorch implementation of the Neural Tangent Kernel (NTK), which replicates the functionality of the scikit-ntk library.

## Overview

The Neural Tangent Kernel is a kernel function that represents the infinite-width limit of a fully-connected ReLU neural network. This implementation is designed to work seamlessly with GPyTorch's Gaussian Process framework.

## Features

- **Feature parity with scikit-ntk**: The implementation produces identical results to the scikit-ntk library
- **GPU-compatible**: Built on PyTorch, automatically supports GPU acceleration when available
- **Batch processing**: Supports batched inputs for efficient computation
- **Learnable parameters**: The bias parameter is learnable during training
- **Priors support**: Supports GPyTorch priors on the bias parameter
- **Active dimensions**: Supports computing the kernel on a subset of input dimensions

## Installation

The kernel requires GPyTorch and PyTorch to be installed:

```bash
pip install gpytorch torch
```

## Usage

### Basic Usage

```python
import torch
from gpytorch_ntk import NeuralTangentKernel

# Create the kernel
ntk = NeuralTangentKernel(depth=3, bias=0.1)

# Create some data
X = torch.randn(100, 5)

# Compute the kernel matrix
K = ntk(X, X)
```

### With Gaussian Process

```python
import torch
import gpytorch
from gpytorch_ntk import NeuralTangentKernel

# Create kernel
ntk = NeuralTangentKernel(depth=3, bias=0.1)

# Create GP model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(ntk)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Train the model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(X, y, likelihood)
# ... training code ...
```

## Parameters

- `depth` (int, >= 1): Number of layers in the neural network
- `bias` (float, >= 0): Bias parameter of the network
- `bias_prior` (Prior, optional): Prior distribution for the bias parameter
- `bias_constraint` (Interval, optional): Constraint for the bias parameter
- `active_dims` (tuple, optional): Indices of dimensions to use for computation
- `batch_shape` (torch.Size, optional): Batch shape for batch processing

## Mathematical Formulation

The kernel is computed using the recursive formula from Jacot et al. (2018):

```
K^0(x1, x2) = x1 · x2 + σ_b^2
K^{l+1}(x1, x2) = (c / 2π) * (λ * (π - arccos(λ)) + √(1 - λ^2)) * √(Σ1 * Σ2) + σ_b^2
K(x1, x2) = 1/(L+1) * ∑_{l=0}^L K^l(x1, x2)
```

where:
- λ = (x1 · x2) / √(Σ1 * Σ2)
- Σ1 = x1 · x1 (squared norm)
- Σ2 = x2 · x2 (squared norm)
- c = 2 (constant for ReLU activation)
- σ_b = bias parameter
- L = depth parameter

## Testing

Run the test suite to verify feature parity with scikit-ntk:

```bash
pytest tests/test_gpytorch_kernel.py
```

## GPU Support

The implementation is fully GPU-compatible. Simply move your tensors to GPU:

```python
ntk = NeuralTangentKernel(depth=3, bias=0.1).cuda()
X = torch.randn(100, 5).cuda()
K = ntk(X, X)  # Computed on GPU
```

## References

- Jacot, A., Gabriel, C., Hongler, M. (2018). Neural Tangent Kernel: Convergence and Generalization in Neural Networks. arXiv:1806.07572
- Lencevicius, R. P. (2022). scikit-ntk: Implementation of the neural tangent kernel for scikit-learn's Gaussian process module.
