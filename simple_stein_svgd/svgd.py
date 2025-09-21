import time
import numpy as np
from dataclasses import dataclass
from .kernels import rbf_kernel, rbf_kernel_stats

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
        Perform one SVGD update.

        Args:
            X: particles (n, d)
            stepsize: step size

        Returns:
            SVGDResult containing:
                - particles: updated particles (n, d)
                - ksd: Kernel Stein Discrepancy computed on the UPDATED particles
                - wall_time: step duration in seconds
        """
        t0 = time.perf_counter()

        n, d = X.shape

        logp_grad = self._grad_log_prob(X)

        K, grad_K = self.kernel(X)

        phi = (K @ logp_grad + np.sum(grad_K, axis=1)) / n

        X_new = X + stepsize * phi

        ksd_val = self._compute_ksd(X_new)

        wall = time.perf_counter() - t0
        return SVGDResult(particles=X_new, ksd=ksd_val, wall_time=wall)

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

    def _compute_ksd(self, X):
        """
        Compute Kernel Stein Discrepancy (V-statistic) for current particles X.

        Currently implemented for the default RBF kernel. If a custom kernel is
        passed in the constructor, this will raise NotImplementedError.
        """
        from .kernels import rbf_kernel_stats

        if self.kernel is not rbf_kernel:
            raise NotImplementedError("KSD is currently implemented only for the default RBF kernel.")

        n, d = X.shape
        score = self._grad_log_prob(X)  # (n, d)

        # Get RBF stats needed for the closed-form Stein kernel terms
        K, grad_K_xi, h, sq_dist = rbf_kernel_stats(X)  # grad wrt x_i
        grad_K_xj = -grad_K_xi  # (n, n, d)

        # 1) s_i^T K_ij s_j
        term1_sum = np.sum((score @ score.T) * K)

        # 2) s_i^T ∇_{x_j} k_ij
        term2_sum = np.einsum('id,ijd->', score, grad_K_xj)

        # 3) s_j^T ∇_{x_i} k_ij
        term3_sum = np.einsum('jd,ijd->', score, grad_K_xi)

        # 4) tr(∇_{x_i}∇_{x_j} k_ij) = (2d/h - 4||x_i-x_j||^2 / h^2) * K_ij
        trace_ij = (2.0 * d / h - 4.0 * sq_dist / (h * h)) * K
        term4_sum = np.sum(trace_ij)

        u_sum = term1_sum + term2_sum + term3_sum + term4_sum
        ksd2 = u_sum / (n * n)
        ksd = float(np.sqrt(max(ksd2, 0.0)))
        return ksd


@dataclass
class SVGDResult:
    particles: np.ndarray
    ksd: float
    wall_time: float
