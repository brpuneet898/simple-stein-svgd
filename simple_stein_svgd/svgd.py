import numpy as np
from .kernels import rbf_kernel

class SVGD:
    """
    Stein Variational Gradient Descent (SVGD)
    """
    def __init__(self, log_prob, kernel=rbf_kernel):
        """
        Args:
            log_prob: function that takes X (n, d) and returns log density
            kernel: kernel function, default is RBF
        """
        self.log_prob = log_prob
        self.kernel = kernel

    def update(self, X, stepsize=0.1):
        """
        Perform one SVGD update
        Args:
            X: particles (n, d)
            stepsize: step size
        Returns:
            updated particles
        """
        n, d = X.shape
        logp_grad = self._grad_log_prob(X)

        K, grad_K = self.kernel(X)

        phi = (K @ logp_grad + np.sum(grad_K, axis=1)) / n
        return X + stepsize * phi

    def _grad_log_prob(self, X, eps=1e-4):
        """
        Numerical gradient of log_prob
        """
        n, d = X.shape
        grad = np.zeros((n, d))
        for i in range(d):
            shift = np.zeros(d)
            shift[i] = eps
            grad[:, i] = (self.log_prob(X + shift) - self.log_prob(X - shift)) / (2 * eps)
        return grad
