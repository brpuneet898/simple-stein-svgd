import time
import numpy as np
from dataclasses import dataclass
from typing import Union, Callable

from .kernels import (
    get_kernel, list_kernels,
    rbf_kernel_factory, bandwidth_median
)

Array = np.ndarray
KernelLike = Union[str, Callable[[Array], tuple]]

class SVGD:
    """
    Stein Variational Gradient Descent (SVGD)
    """
    def __init__(self,
                 log_prob: Callable[[Array], Array],
                 kernel: KernelLike = "rbf",
                 bandwidth_rule: Callable[[Array], float] = bandwidth_median):
        """
        Args:
            log_prob: function taking X (n,d) and returning log density at each point (shape (n,))
                      OR returning a scalar log-density for each row; numerical grads are taken.
            kernel: kernel name ('rbf'|'imq'|'linear' or custom callable) from the registry.
                    If a callable is passed, it must return (K, grad_K_xi, grad_K_xj, trace_xixj).
            bandwidth_rule: only used by some built-in kernels when 'kernel' is a string and
                           the factory can consume a rule (e.g., RBF/IMQ). Ignored otherwise.
        """
        self.log_prob = log_prob
        if isinstance(kernel, str):
            kname = kernel.lower()
            if kname == "rbf":
                self.kernel = rbf_kernel_factory(bandwidth_rule)
            else:
                self.kernel = get_kernel(kernel)
        else:
            self.kernel = kernel

    def update(self, X: Array, stepsize: float = 0.1):
        """
        Perform one SVGD update.

        Returns:
            SVGDResult(particles, ksd, wall_time)
        """
        t0 = time.perf_counter()

        n, d = X.shape

        logp_grad = self._grad_log_prob(X)  

        K, grad_K_xi, _, _ = self.kernel(X)  

        phi = (K @ logp_grad + np.sum(grad_K_xi, axis=1)) / n

        X_new = X + stepsize * phi

        ksd_val = self._compute_ksd(X_new)

        wall = time.perf_counter() - t0
        return SVGDResult(particles=X_new, ksd=ksd_val, wall_time=wall)

    def _grad_log_prob(self, X: Array, eps: float = 1e-4) -> Array:
        """
        Numerical gradient of log_prob.
        Supports log_prob that returns (n,) or (n,1) or scalar per-row via broadcasting.
        """
        n, d = X.shape
        grad = np.zeros((n, d), dtype=float)
        for i in range(d):
            shift = np.zeros(d, dtype=float)
            shift[i] = eps
            lp_plus = self.log_prob(X + shift)
            lp_minus = self.log_prob(X - shift)
            grad[:, i] = (lp_plus - lp_minus).reshape(n) / (2 * eps)
        return grad

    def _compute_ksd(self, X: Array) -> float:
        """
        Generic Kernel Stein Discrepancy (V-statistic) for the current kernel.
        Works for any kernel that returns (K, grad_K_xi, grad_K_xj, trace_xixj).
        """
        n, d = X.shape
        s = self._grad_log_prob(X) 
        K, grad_K_xi, grad_K_xj, trace_xixj = self.kernel(X)

        term1 = np.sum((s @ s.T) * K)

        term2 = np.einsum('id,ijd->', s, grad_K_xj)

        term3 = np.einsum('jd,ijd->', s, grad_K_xi)

        term4 = np.sum(trace_xixj)

        u_sum = term1 + term2 + term3 + term4
        ksd2 = u_sum / (n * n)
        return float(np.sqrt(max(ksd2, 0.0)))

@dataclass
class SVGDResult:
    particles: np.ndarray
    ksd: float
    wall_time: float
