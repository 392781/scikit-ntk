"""
Neural Tangent Kernel for GPyTorch.

This module implements the Neural Tangent Kernel as a custom GPyTorch kernel.
The implementation is based on the scikit-ntk library by Ronaldas P Lencevicius.

Reference:
    Lencevicius, R. P. (2022). scikit-ntk: Implementation of the neural tangent kernel
    for scikit-learn's Gaussian process module.
"""

import math
import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
from linear_operator import LinearOperator


class NeuralTangentKernel(Kernel):
    r"""
    Neural tangent kernel for GPyTorch.

    The neural tangent kernel can be used to represent a fully connected
    infinite width ReLU activated neural network. It is parameterized with
    the network's depth and bias.

    The kernel is computed as described in the paper:
    "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"
    by Jacot, Gabriel, and Hongler (2018).

    The implementation follows the recursive formula from the paper:
    
    .. math::
        \begin{align*}
            K^0(x_1, x_2) &= x_1 \cdot x_2 + \sigma_b^2 \\
            K^{l+1}(x_1, x_2) &= \frac{c}{2\pi} \left( \Sigma \left( \pi - \arccos\left(\frac{\Sigma}{\sqrt{\Sigma_1 \Sigma_2}}\right)\right) + \sqrt{1 - \frac{\Sigma^2}{\Sigma_1 \Sigma_2}} \right) + \sigma_b^2 \\
            K(x_1, x_2) &= \frac{1}{L+1} \sum_{l=0}^L K^l(x_1, x_2)
        \end{align*}

    where:
    - :math:`\Sigma = x_1 \cdot x_2` (the dot product)
    - :math:`\Sigma_1 = x_1 \cdot x_1` (the squared norm of x1)
    - :math:`\Sigma_2 = x_2 \cdot x_2` (the squared norm of x2)
    - :math:`c = 2` (constant for ReLU activation)
    - :math:`\sigma_b` is the bias parameter
    - :math:`L` is the depth parameter

    :param depth: Number of layers in the neural network (>= 1)
    :param bias: Bias parameter (>= 0)
    :param bias_prior: Prior over the bias parameter (default: None)
    :param bias_constraint: Constraint on the bias parameter (default: None, meaning no constraint)
    :param active_dims: Dimensions to use for computation (default: None)
    :param batch_shape: Batch shape for batch processing (default: None)

    Example:
        >>> import torch
        >>> from gpytorch_ntk import NeuralTangentKernel
        >>> kernel = NeuralTangentKernel(depth=3, bias=0.1)
        >>> x = torch.randn(10, 5)
        >>> covar = kernel(x, x)
    """

    has_lengthscale = False

    def __init__(
        self,
        depth: int,
        bias: float = 0.1,
        bias_prior: Prior | None = None,
        bias_constraint: Interval | None = None,
        active_dims: tuple[int, ...] | None = None,
        batch_shape: torch.Size | None = None,
        **kwargs,
    ):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if bias < 0:
            raise ValueError("bias must be >= 0")
        
        super().__init__(active_dims=active_dims, batch_shape=batch_shape, **kwargs)
        
        self.depth = depth
        self.c = 2.0  # Constant for ReLU activation
        
        # Register bias parameter
        # We want the bias to be used directly without transformation by default
        # Only apply constraint if explicitly provided
        self.raw_bias = torch.nn.Parameter(torch.tensor(bias))
        
        # Register the parameter
        self.register_parameter(name="raw_bias", parameter=self.raw_bias)
        
        # Only register constraint if explicitly provided
        self._bias_constraint = bias_constraint
        if bias_constraint is not None:
            self.register_constraint("raw_bias", bias_constraint)
        
        if bias_prior is not None:
            if not isinstance(bias_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(bias_prior).__name__)
            self.register_prior(
                "bias_prior",
                bias_prior,
                lambda m: m.bias,
                lambda m, v: m._set_bias(v),
            )

    @property
    def bias(self) -> torch.Tensor:
        """The bias parameter (transformed through constraint if applicable)."""
        if self._bias_constraint is not None:
            return self._bias_constraint.transform(self.raw_bias)
        return self.raw_bias

    @bias.setter
    def bias(self, value: float | torch.Tensor) -> None:
        self._set_bias(value)

    def _set_bias(self, value: float | torch.Tensor) -> None:
        """Set the bias parameter."""
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_bias)
        
        # If there's a constraint, apply inverse transform
        if self._bias_constraint is not None:
            value = self._bias_constraint.inverse_transform(value)
        
        self.initialize(raw_bias=value)

    @property
    def is_stationary(self) -> bool:
        """The NTK is not a stationary kernel."""
        return False

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor | LinearOperator:
        r"""
        Computes the covariance between x1 and x2 using the Neural Tangent Kernel.

        :param x1: First set of data (... x N x D)
        :param x2: Second set of data (... x M x D)
        :param diag: If True, compute only the diagonal (x1 must equal x2)
        :param last_dim_is_batch: If True, treat last dimension as batch
        :return: Covariance matrix of shape (... x N x M) or diagonal of shape (... x N)
        """
        # Handle active dimensions
        if self.active_dims is not None:
            x1 = x1.index_select(-1, self.active_dims)
            x2 = x2.index_select(-1, self.active_dims)
        
        # For diagonal computation
        if diag:
            return self._compute_diag(x1)
        
        # Check if x1 and x2 are the same
        # Following scikit-ntk, we need to augment when they're different
        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Same inputs, compute directly
            return self._compute_kernel_matrix(x1, x2)
        else:
            # Different inputs, use augmented approach
            return self._compute_kernel_matrix_augmented(x1, x2)

    def _compute_kernel_matrix(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the full kernel matrix between x1 and x2 (when x1 == x2).
        
        :param x1: First set of data
        :param x2: Second set of data (same as x1)
        :return: Kernel matrix
        """
        # Ensure 2D tensors
        if x1.dim() > 2:
            # Flatten batch dimensions
            x1_flat = x1.reshape(-1, x1.size(-1))
            x2_flat = x2.reshape(-1, x2.size(-1))
            kernel_flat = self._compute_kernel_matrix_2d(x1_flat, x2_flat)
            # Reshape back to include batch dimensions
            batch_shape = x1.shape[:-2]
            n = x1.shape[-2]
            return kernel_flat.reshape(*batch_shape, n, n)
        
        return self._compute_kernel_matrix_2d(x1, x2)

    def _compute_kernel_matrix_augmented(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute kernel matrix between different x1 and x2 using augmented approach.
        This follows the scikit-ntk implementation.
        
        :param x1: First set of data (N x D)
        :param x2: Second set of data (M x D)
        :return: Kernel matrix (N x M)
        """
        # Flatten to 2D if needed
        if x1.dim() > 2:
            x1_flat = x1.reshape(-1, x1.size(-1))
            x2_flat = x2.reshape(-1, x2.size(-1))
            kernel_flat = self._compute_kernel_matrix_augmented_2d(x1_flat, x2_flat)
            # Reshape back
            batch_shape = x1.shape[:-2]
            n = x1.shape[-2]
            m = x2.shape[-2]
            return kernel_flat.reshape(*batch_shape, n, m)
        
        return self._compute_kernel_matrix_augmented_2d(x1, x2)

    def _compute_kernel_matrix_augmented_2d(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute kernel matrix for 2D tensors using augmented approach.
        
        :param x1: First set of data (N x D)
        :param x2: Second set of data (M x D)
        :return: Kernel matrix (N x M)
        """
        # Store original shapes
        x1_shape = x1.shape[0]
        x2_shape = x2.shape[0]
        
        # Concatenate x1 and x2
        augmented = torch.cat([x1, x2], dim=0)
        
        # Compute full kernel on augmented matrix
        K_full = self._compute_kernel_matrix_2d(augmented, augmented)
        
        # Extract the cross-kernel K(x1, x2)
        return K_full[:x1_shape, x1_shape:(x1_shape + x2_shape)]

    def _compute_kernel_matrix_2d(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the kernel matrix for 2D input tensors.
        Assumes x1 and x2 are the same or we want K(x1, x2) where x1 == x2.
        
        :param x1: First set of data (N x D)
        :param x2: Second set of data (N x D, same as x1)
        :return: Kernel matrix (N x N)
        """
        # Compute the dot product matrix: Σ = x1 @ x2.T
        sigma_mat = torch.matmul(x1, x2.transpose(-2, -1))
        
        # Initialize K with K^0 = Σ + bias^2
        bias_sq = self.bias ** 2
        K = sigma_mat + bias_sq
        
        # Iterate through layers 1 to depth
        for dep in range(1, self.depth + 1):
            # Compute diagonal elements (Σ_1 and Σ_2)
            # Since x1 == x2, we can use the diagonal of sigma_mat
            diag = torch.diag(sigma_mat)
            
            # Compute denominator: sqrt(outer(diag, diag))
            diag_outer = torch.outer(diag, diag)
            denominator = torch.clamp(
                torch.sqrt(diag_outer),
                min=1e-10,
                max=float('inf')
            )
            
            # Compute λ = Σ / denominator
            lambda_mat = sigma_mat / denominator
            lambda_mat = torch.nan_to_num(lambda_mat, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clip to [-1, 1] for arccos
            lambda_mat = torch.clamp(lambda_mat, min=-1.0, max=1.0)
            
            # Compute the arc cosine term
            arccos_term = torch.pi - torch.arccos(lambda_mat)
            
            # Compute the new sigma matrix for this layer
            # Σ^{l+1} = (c / (2π)) * (λ * arccos_term + sqrt(1 - λ^2)) * denominator
            sqrt_term = torch.sqrt(1.0 - lambda_mat ** 2)
            sigma_mat_new = (self.c / (2 * math.pi)) * (
                lambda_mat * arccos_term + sqrt_term
            ) * denominator
            
            # Compute the dot product term for this layer
            # K^{l+1}_dot = (c / (2π)) * (π - arccos(λ))
            K_dot = (self.c / (2 * math.pi)) * arccos_term
            
            # Update K: K = K * K_dot + Σ^{l+1} + bias^2
            K = K * K_dot + sigma_mat_new + bias_sq
            
            # Update sigma_mat for next iteration
            sigma_mat = sigma_mat_new
        
        # Apply the final scaling
        scalar = 1.0 / ((self.depth + 1) * (bias_sq + 1))
        return scalar * K

    def _compute_diag(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the diagonal of the kernel matrix.
        
        :param x: Input data (N x D)
        :return: Diagonal elements (N,)
        """
        if x.dim() > 2:
            # Flatten batch dimensions
            x_flat = x.reshape(-1, x.size(-1))
            diag_flat = self._compute_diag_2d(x_flat)
            # Reshape back
            batch_shape = x.shape[:-2]
            n = x.shape[-2]
            return diag_flat.reshape(*batch_shape, n)
        
        return self._compute_diag_2d(x)

    def _compute_diag_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the diagonal for 2D input.
        
        :param x: Input data (N x D)
        :return: Diagonal elements (N,)
        """
        # Compute the full kernel matrix and extract diagonal
        K = self._compute_kernel_matrix_2d(x, x)
        return torch.diag(K)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(depth={self.depth}, bias={self.bias.item():.3f})"
