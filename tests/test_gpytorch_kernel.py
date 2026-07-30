"""
Tests for GPyTorch Neural Tangent Kernel implementation.

These tests verify feature parity with the scikit-ntk implementation.
"""

import pytest
import numpy as np
import torch

from skntk import NeuralTangentKernel as SklearnNTK
from gpytorch_ntk import NeuralTangentKernel as GPyTorchNTK


# Test data
np.random.seed(42)
torch.manual_seed(42)

# Create test inputs
X_np = np.random.RandomState(0).normal(0, 1, (5, 2))
Y_np = np.random.RandomState(0).normal(0, 1, (6, 2))

# Convert to torch tensors
X_torch = torch.from_numpy(X_np).float()
Y_torch = torch.from_numpy(Y_np).float()


@pytest.fixture
def sklearn_ntk():
    """Create a scikit-ntk kernel instance."""
    return SklearnNTK(depth=3, bias=0.01, bias_bounds=(1e-5, 1e3))


@pytest.fixture
def gpytorch_ntk():
    """Create a GPyTorch NTK kernel instance."""
    return GPyTorchNTK(depth=3, bias=0.01)


def test_kernel_values_match(sklearn_ntk, gpytorch_ntk):
    """Test that kernel values match between implementations."""
    # Compute scikit-ntk kernel
    K_sklearn = sklearn_ntk(X_np)
    
    # Compute GPyTorch kernel
    with torch.no_grad():
        K_gpytorch = gpytorch_ntk(X_torch, X_torch).numpy()
    
    # Check shapes match
    assert K_sklearn.shape == K_gpytorch.shape
    
    # Check values are close (allowing for small numerical differences)
    np.testing.assert_allclose(K_sklearn, K_gpytorch, rtol=1e-5, atol=1e-7)


def test_kernel_cross_values_match(sklearn_ntk, gpytorch_ntk):
    """Test that cross-kernel values (K(X, Y)) match between implementations."""
    # Compute scikit-ntk kernel
    K_sklearn = sklearn_ntk(X_np, Y_np)
    
    # Compute GPyTorch kernel
    with torch.no_grad():
        K_gpytorch = gpytorch_ntk(X_torch, Y_torch).numpy()
    
    # Check shapes match
    assert K_sklearn.shape == K_gpytorch.shape
    
    # Check values are close
    np.testing.assert_allclose(K_sklearn, K_gpytorch, rtol=1e-5, atol=1e-7)


def test_kernel_diagonal(sklearn_ntk, gpytorch_ntk):
    """Test that diagonal values match."""
    # Compute scikit-ntk diagonal
    diag_sklearn = sklearn_ntk.diag(X_np)
    
    # Compute GPyTorch diagonal
    with torch.no_grad():
        diag_gpytorch = gpytorch_ntk(X_torch, X_torch, diag=True).numpy()
    
    # Check shapes match
    assert diag_sklearn.shape == diag_gpytorch.shape
    
    # Check values are close
    np.testing.assert_allclose(diag_sklearn, diag_gpytorch, rtol=1e-5, atol=1e-7)


def test_different_depths():
    """Test kernels with different depth values."""
    depths = [1, 2, 3, 5]
    bias = 0.1
    
    for depth in depths:
        sklearn_ntk = SklearnNTK(depth=depth, bias=bias, bias_bounds=(1e-5, 1e3))
        gpytorch_ntk = GPyTorchNTK(depth=depth, bias=bias)
        
        K_sklearn = sklearn_ntk(X_np)
        with torch.no_grad():
            K_gpytorch = gpytorch_ntk(X_torch, X_torch).numpy()
        
        np.testing.assert_allclose(K_sklearn, K_gpytorch, rtol=1e-5, atol=1e-7)


def test_different_biases():
    """Test kernels with different bias values."""
    depth = 3
    biases = [0.0, 0.01, 0.1, 1.0]
    
    for bias in biases:
        sklearn_ntk = SklearnNTK(depth=depth, bias=bias, bias_bounds=(1e-5, 1e3))
        gpytorch_ntk = GPyTorchNTK(depth=depth, bias=bias)
        
        K_sklearn = sklearn_ntk(X_np)
        with torch.no_grad():
            K_gpytorch = gpytorch_ntk(X_torch, X_torch).numpy()
        
        np.testing.assert_allclose(K_sklearn, K_gpytorch, rtol=1e-5, atol=1e-7)


def test_kernel_symmetry(gpytorch_ntk):
    """Test that the kernel matrix is symmetric."""
    with torch.no_grad():
        K = gpytorch_ntk(X_torch, X_torch).numpy()
    
    # Check symmetry
    np.testing.assert_allclose(K, K.T, rtol=1e-10, atol=1e-10)


def test_kernel_positive_definite(gpytorch_ntk):
    """Test that the kernel matrix is positive semi-definite."""
    with torch.no_grad():
        K = gpytorch_ntk(X_torch, X_torch).numpy()
    
    # Check positive semi-definite (all eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(eigenvalues >= -1e-10), "Kernel matrix is not positive semi-definite"


def test_known_values_depth_1():
    """Test known values for depth=1 kernel."""
    # For depth=1, the kernel has a specific form
    depth = 1
    bias = 0.0
    
    sklearn_ntk = SklearnNTK(depth=depth, bias=bias, bias_bounds=(1e-5, 1e3))
    gpytorch_ntk = GPyTorchNTK(depth=depth, bias=bias)
    
    # Simple test case
    X_simple = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    X_simple_torch = torch.from_numpy(X_simple).float()
    
    K_sklearn = sklearn_ntk(X_simple)
    with torch.no_grad():
        K_gpytorch = gpytorch_ntk(X_simple_torch, X_simple_torch).numpy()
    
    np.testing.assert_allclose(K_sklearn, K_gpytorch, rtol=1e-5, atol=1e-7)


def test_known_values_depth_2():
    """Test known values for depth=2 kernel."""
    depth = 2
    bias = 0.1
    
    sklearn_ntk = SklearnNTK(depth=depth, bias=bias, bias_bounds=(1e-5, 1e3))
    gpytorch_ntk = GPyTorchNTK(depth=depth, bias=bias)
    
    # Simple test case
    X_simple = np.array([[1.0, 0.0], [0.0, 1.0]])
    X_simple_torch = torch.from_numpy(X_simple).float()
    
    K_sklearn = sklearn_ntk(X_simple)
    with torch.no_grad():
        K_gpytorch = gpytorch_ntk(X_simple_torch, X_simple_torch).numpy()
    
    np.testing.assert_allclose(K_sklearn, K_gpytorch, rtol=1e-5, atol=1e-7)


def test_batch_processing():
    """Test batch processing with GPyTorch kernel."""
    gpytorch_ntk = GPyTorchNTK(depth=3, bias=0.1)
    
    # Create batch input
    batch_x = torch.randn(2, 5, 3)  # 2 batches, 5 samples, 3 features
    
    # Compute kernel
    with torch.no_grad():
        K = gpytorch_ntk(batch_x, batch_x)
    
    # Check shape
    assert K.shape == (2, 5, 5), f"Expected shape (2, 5, 5), got {K.shape}"


def test_repr():
    """Test string representation."""
    nt = GPyTorchNTK(depth=3, bias=0.1)
    repr_str = repr(nt)
    assert "NeuralTangentKernel" in repr_str
    assert "depth=3" in repr_str
    assert "bias=0.100" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
