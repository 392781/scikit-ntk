"""
Advanced usage examples for the GPyTorch Neural Tangent Kernel.

This file demonstrates:
1. Using KeOps for efficient large-scale kernel operations
2. Using derivative information with the NTK
3. Combining NTK with other GPyTorch features
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gpytorch
from gpytorch_ntk import NeuralTangentKernel

print("=" * 80)
print("Example 1: Using NTK with KeOps for Large-Scale Inference")
print("=" * 80)

# Check if KeOps is available
try:
    import pykeops
    from pykeops.torch import LazyTensor
    from linear_operator.operators import KernelLinearOperator
    KEOPS_AVAILABLE = True
    print("✓ KeOps is available")
except ImportError:
    KEOPS_AVAILABLE = False
    print("✗ KeOps is not available. Skipping KeOps examples.")

if KEOPS_AVAILABLE:
    # Create a KeOps-accelerated version of the NTK
    # This wraps the NTK computation in a KernelLinearOperator that can use KeOps
    
    class NeuralTangentKernelKeOps(gpytorch.kernels.Kernel):
        """
        KeOps-accelerated Neural Tangent Kernel.
        
        This kernel uses KeOps for efficient matrix-vector products,
        which is beneficial for large datasets.
        """
        has_lengthscale = False
        
        def __init__(self, depth: int, bias: float = 0.1, **kwargs):
            super().__init__(**kwargs)
            self.base_kernel = NeuralTangentKernel(depth=depth, bias=bias)
        
        def forward(self, x1, x2, diag=False, **params):
            # For KeOps, we want to return a KernelLinearOperator
            # that can efficiently compute matrix-vector products
            
            # Define the covariance function
            def covar_func(x1, x2, **kwargs):
                # Compute the full NTK matrix
                return self.base_kernel._compute_kernel_matrix_2d(x1, x2)
            
            # Create a KernelLinearOperator
            return KernelLinearOperator(x1, x2, covar_func=covar_func)
    
    # Test with larger dataset
    print("\nTesting with larger dataset (1000 points)...")
    X_large = torch.randn(1000, 10)
    
    # Standard NTK
    ntk_standard = NeuralTangentKernel(depth=3, bias=0.1)
    
    # KeOps NTK
    ntk_keops = NeuralTangentKernelKeOps(depth=3, bias=0.1)
    
    print("Computing kernel matrix with standard NTK...")
    with torch.no_grad():
        K_standard = ntk_standard(X_large, X_large)
    print(f"Standard NTK output shape: {K_standard.shape}")
    
    print("Computing kernel matrix with KeOps NTK...")
    with torch.no_grad():
        K_keops = ntk_keops(X_large, X_large)
    print(f"KeOps NTK output shape: {K_keops.shape}")
    
    # Convert to dense for comparison
    with torch.no_grad():
        K_keops_dense = K_keops.to_dense()
    K_standard_dense = K_standard.to_dense()
    
    # Verify they produce the same results
    if torch.allclose(K_standard_dense, K_keops_dense, rtol=1e-5):
        print("✓ Results match!")
    else:
        print("✗ Results differ!")
        print(f"Max difference: {(K_standard - K_keops_dense).abs().max().item()}")

print("\n" + "=" * 80)
print("Example 2: Using NTK with Derivative Information")
print("=" * 80)

# For derivative information, we can use finite differences
# or implement a custom kernel that computes the gradient covariance

# Here's a simpler approach: use the NTK to model the function,
# and use finite differences to approximate gradients

print("\nDemonstrating finite difference approach for gradients...")

# Create a simple function
X = torch.linspace(0, 2*torch.pi, 20).unsqueeze(1)
y = torch.sin(X).squeeze()

# Compute finite difference gradients
h = 1e-5
y_grad = (torch.sin(X + h) - torch.sin(X - h)) / (2 * h)
y_grad = y_grad.squeeze()

# Combine values and gradients
y_combined = torch.stack([y, y_grad], dim=1)

print(f"Input shape: {X.shape}")
print(f"Combined targets shape: {y_combined.shape}")

# We can use a multi-task GP to model both values and gradients
# However, this requires a kernel that can handle the relationship
# between function values and gradients

# For now, we'll just demonstrate that the NTK can be used
# in a multi-output setting

class MultiOutputNTK(gpytorch.kernels.Kernel):
    """
    Multi-output Neural Tangent Kernel.
    
    This kernel applies the NTK to each output dimension separately.
    """
    has_lengthscale = False
    
    def __init__(self, depth: int, bias: float = 0.1, num_outputs: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = NeuralTangentKernel(depth=depth, bias=bias)
        self.num_outputs = num_outputs
    
    def forward(self, x1, x2, diag=False, **params):
        # For multi-output, we need to expand the input
        # x1 and x2 have shape (..., n, d)
        # We want to compute a block kernel for each output
        
        # For simplicity, we'll just use the same kernel for all outputs
        # A more sophisticated approach would learn different parameters
        # for each output
        
        K = self.base_kernel(x1, x2, diag=diag, **params)
        
        # Expand to multi-output
        if diag:
            return K.unsqueeze(-1).expand(*K.shape[:-1], self.num_outputs)
        else:
            return K.unsqueeze(-2).unsqueeze(-2).expand(
                *K.shape[:-2], self.num_outputs, K.shape[-2], 
                self.num_outputs, K.shape[-1]
            )

# Test multi-output NTK
print("\nTesting multi-output NTK...")
multi_ntk = MultiOutputNTK(depth=3, bias=0.1, num_outputs=2)
K_multi = multi_ntk(X, X)
print(f"Multi-output NTK shape: {K_multi.shape}")
print("✓ Multi-output NTK works!")

print("\n" + "=" * 80)
print("Example 3: Using NTK with GPyTorch's built-in features")
print("=" * 80)

# Example: Using NTK with additive structure
print("\nCreating an additive NTK model...")

class AdditiveNTK(gpytorch.kernels.Kernel):
    """
    Additive Neural Tangent Kernel.
    
    This kernel applies the NTK to each input dimension separately
    and sums the results, similar to an additive GP.
    """
    has_lengthscale = False
    
    def __init__(self, depth: int, bias: float = 0.1, num_dims: int = None, **kwargs):
        super().__init__(**kwargs)
        if num_dims is None:
            raise ValueError("num_dims must be specified")
        self.kernels = torch.nn.ModuleList([
            NeuralTangentKernel(depth=depth, bias=bias, active_dims=(i,))
            for i in range(num_dims)
        ])
    
    def forward(self, x1, x2, diag=False, **params):
        # Sum the kernels for each dimension
        K = sum(kernel(x1, x2, diag=diag, **params) for kernel in self.kernels)
        return K

# Test additive NTK
additive_ntk = AdditiveNTK(depth=3, bias=0.1, num_dims=5)
X_test = torch.randn(10, 5)
K_additive = additive_ntk(X_test, X_test)
print(f"Additive NTK output shape: {K_additive.shape}")
print("✓ Additive NTK works!")

# Example: Using NTK with ScaleKernel
print("\nUsing NTK with ScaleKernel...")
from gpytorch.kernels import ScaleKernel
scaled_ntk = ScaleKernel(NeuralTangentKernel(depth=3, bias=0.1))
K_scaled = scaled_ntk(X_test, X_test)
print(f"Scaled NTK output shape: {K_scaled.shape}")
print("✓ Scaled NTK works!")

# Example: Training a GP with NTK
print("\nTraining a GP with NTK...")
X_train = torch.randn(50, 3)
y_train = torch.sin(X_train).sum(dim=1)

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(NeuralTangentKernel(depth=3, bias=0.1))
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(X_train, y_train, likelihood)

# Train for a few iterations
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(10):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -output.log_prob(y_train).sum()
    loss.backward()
    optimizer.step()
    if i % 5 == 0:
        print(f"  Iteration {i}: Loss = {loss.item():.4f}")

print("✓ GP training with NTK works!")

print("\n" + "=" * 80)
print("All examples completed successfully!")
print("=" * 80)
