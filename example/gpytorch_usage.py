"""
Example usage of the GPyTorch Neural Tangent Kernel.

This demonstrates how to use the NeuralTangentKernel with GPyTorch
for Gaussian Process regression.
"""

import sys
import os
# Add the parent directory to the path so we can import gpytorch_ntk
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gpytorch
from gpytorch_ntk import NeuralTangentKernel

# Set random seed for reproducibility
torch.manual_seed(42)

# Create some sample data
n = 100
d = 5
X = torch.randn(n, d)
y = torch.sin(X.sum(dim=1)) + 0.1 * torch.randn(n)

# Create the Neural Tangent Kernel
ntk = NeuralTangentKernel(depth=3, bias=0.1)

# Create a GP model using the NTK
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(ntk)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Create likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(X, y, likelihood)

# Train the model
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop
for i in range(50):
    optimizer.zero_grad()
    output = model(X)
    loss = -output.log_prob(y).sum()
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(f"Iteration {i}: Loss = {loss.item():.4f}")

# Evaluate the model
model.eval()
likelihood.eval()

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_X = torch.randn(10, d)
    predictions = likelihood(model(test_X))
    print(f"\nTest predictions mean: {predictions.mean}")
    print(f"Test predictions variance: {predictions.variance}")

print("\nModel trained successfully with Neural Tangent Kernel!")
print(f"Kernel: {model.covar_module.base_kernel}")
print(f"Lengthscale: {model.covar_module.lengthscale}")
print(f"Outputscale: {model.covar_module.outputscale}")
