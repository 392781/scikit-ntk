import pytest
import numpy as np

from sklearn.gaussian_process.kernels import _approx_fprime
from numpy.testing import assert_almost_equal
from skntk import NeuralTangentKernel as NTK

X = np.random.RandomState(0).normal(0, 1, (5, 2))
Y = np.random.RandomState(0).normal(0, 1, (6, 2))

ntk = NTK(depth=3, bias=0.01, bias_bounds=(1e-5, 1e3))


def test_kernel_gradient():
    # Compare analytic and numeric gradient of kernels.
    K, K_gradient = ntk(X, eval_gradient=True)

    assert K_gradient.shape[0] == X.shape[0]
    assert K_gradient.shape[1] == X.shape[0]
    assert K_gradient.shape[2] == ntk.theta.shape[0]

    def eval_kernel_for_theta(theta):
        ntk_clone = ntk.clone_with_theta(theta)
        K = ntk_clone(X, eval_gradient=False)
        return K

    K_gradient_approx = _approx_fprime(ntk.theta, eval_kernel_for_theta, 1e-10)
    assert_almost_equal(K_gradient, K_gradient_approx, 4)
