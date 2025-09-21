import numpy as np
from typing import Callable, Dict, Tuple, Optional

# -----------------------------
# Bandwidth (and scale) rules
# -----------------------------

def _pairwise_sqdist(X: np.ndarray) -> np.ndarray:
    return np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)

def bandwidth_median(X: np.ndarray) -> float:
    """Median trick: h = median(||xi-xj||^2) / log(n+1)."""
    n = X.shape[0]
    sq = _pairwise_sqdist(X)
    return float(np.median(sq) / np.log(n + 1.0) if n > 1 else 1.0)

def bandwidth_scott(X: np.ndarray) -> float:
    """
    Scott's rule on per-dim std, then convert to a squared length scale.
    Returns an 'h' used as in exp(-||x-y||^2 / h). We take h = (c * sigma)^2.
    """
    n, d = X.shape
    if n < 2:
        return 1.0
    sigma = np.std(X, axis=0, ddof=1).mean()
    c = n ** (-1.0 / (d + 4))  # Scott factor
    ell = max(c * sigma, 1e-12)
    return float(ell * ell)

def bandwidth_silverman(X: np.ndarray) -> float:
    """Silverman's rule of thumb → convert to squared length scale h."""
    n, d = X.shape
    if n < 2:
        return 1.0
    sigma = np.std(X, axis=0, ddof=1).mean()
    c = (n * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0))
    ell = max(c * sigma, 1e-12)
    return float(ell * ell)

def bandwidth_fixed(value: float) -> Callable[[np.ndarray], float]:
    """Return a rule that always emits the given value."""
    def rule(_X: np.ndarray) -> float:
        return float(value)
    return rule

_BW_RULES: Dict[str, Callable[[np.ndarray], float]] = {
    "median": bandwidth_median,
    "scott": bandwidth_scott,
    "silverman": bandwidth_silverman,
}

def list_bandwidth_rules() -> Dict[str, Callable[[np.ndarray], float]]:
    return dict(_BW_RULES)

# -----------------------------
# Kernel registry
# -----------------------------

KernelFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
_REGISTRY: Dict[str, KernelFn] = {}

def register_kernel(name: str, fn: KernelFn) -> None:
    key = name.lower().strip()
    _REGISTRY[key] = fn

def get_kernel(name_or_fn) -> KernelFn:
    if callable(name_or_fn):
        return name_or_fn
    key = str(name_or_fn).lower().strip()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown kernel '{name_or_fn}'. Known: {list(_REGISTRY)}")
    return _REGISTRY[key]

def list_kernels() -> Dict[str, KernelFn]:
    return dict(_REGISTRY)

# -----------------------------
# RBF (Gaussian) kernel
# k(x,y) = exp(-||x-y||^2 / h)
# -----------------------------

def rbf_kernel_factory(bw_rule: Optional[Callable[[np.ndarray], float]] = None) -> KernelFn:
    rule = bw_rule or bandwidth_median

    def kfn(X: np.ndarray):
        n, d = X.shape
        sq = _pairwise_sqdist(X)
        h = rule(X)
        K = np.exp(-sq / h)

        diff = X[:, None, :] - X[None, :, :]
        grad_K_xi = (-2.0 / h) * diff * K[:, :, None]
        grad_K_xj = -grad_K_xi  

        trace_xixj = (2.0 * d / h - 4.0 * sq / (h * h)) * K
        return K, grad_K_xi, grad_K_xj, trace_xixj

    return kfn

def rbf_kernel(X: np.ndarray, h: float = -1):
    """
    Legacy RBF that matches previous API. Kept for backward compatibility.
    """
    rule = (bandwidth_median if h <= 0 else bandwidth_fixed(h))
    return rbf_kernel_factory(rule)(X)

# -----------------------------
# IMQ kernel
# k(x,y) = (alpha + ||x-y||^2)^(-beta)   with alpha>0, beta>0
# -----------------------------

def imq_kernel_factory(alpha_rule: Optional[Callable[[np.ndarray], float]] = None,
                       beta: float = 0.5) -> KernelFn:
    arule = alpha_rule or bandwidth_median  

    def kfn(X: np.ndarray):
        n, d = X.shape
        sq = _pairwise_sqdist(X)
        alpha = arule(X)
        base = (alpha + sq)
        K = base ** (-beta)

        diff = X[:, None, :] - X[None, :, :]
        grad_factor = (-2.0 * beta) * (base ** (-beta - 1.0))
        grad_K_xi = diff * grad_factor[:, :, None]
        grad_K_xj = -grad_K_xi

        s = sq
        term = 2.0 * beta * d * (base ** (-beta - 1.0)) - \
               4.0 * beta * (beta + 1.0) * s * (base ** (-beta - 2.0))
        trace_xixj = term
        return K, grad_K_xi, grad_K_xj, trace_xixj

    return kfn

def imq_kernel(X: np.ndarray, alpha: float = -1.0, beta: float = 0.5):
    rule = (bandwidth_median if alpha <= 0 else bandwidth_fixed(alpha))
    return imq_kernel_factory(rule, beta)(X)

# -----------------------------
# Linear kernel
# k(x,y) = x^T y
# -----------------------------

def linear_kernel_factory() -> KernelFn:
    def kfn(X: np.ndarray):
        n, d = X.shape
        K = X @ X.T  
        grad_K_xi = np.broadcast_to(X[None, :, :], (n, n, d))
        grad_K_xj = np.broadcast_to(X[:, None, :], (n, n, d))
        trace_xixj = np.full((n, n), float(d))
        return K, grad_K_xi, grad_K_xj, trace_xixj
    return kfn

def linear_kernel(X: np.ndarray):
    return linear_kernel_factory()(X)

# -----------------------------
# Register built-ins (with default rules)
# -----------------------------

register_kernel("rbf", rbf_kernel_factory())
register_kernel("imq", imq_kernel_factory(beta=0.5))
register_kernel("linear", linear_kernel_factory())

def rbf_kernel_stats(X, h=-1):
    """
    For back-compat with prior API that returned (K, grad_K_xi, h, sq_dist).
    """
    n, d = X.shape
    sq = _pairwise_sqdist(X)
    rule = (bandwidth_median if h <= 0 else bandwidth_fixed(h))
    hh = rule(X)
    K = np.exp(-sq / hh)
    diff = X[:, None, :] - X[None, :, :]
    grad_K_xi = (-2.0 / hh) * diff * K[:, :, None]
    return K, grad_K_xi, hh, sq
