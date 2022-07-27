from skntk import NeuralTangentKernel as NTK
from sklearn.datasets import make_friedman1
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

ntk = NTK(3)
X, y = make_friedman1()
gp = GPR(kernel=ntk)

test = gp.fit(X, y)
print(gp.kernel_)
