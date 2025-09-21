"""
Simple-Stein-SVGD: A minimal Python package for Stein Variational Gradient Descent
"""

from .svgd import SVGD, SVGDResult
from .kernels import (
    register_kernel, get_kernel, list_kernels,
    rbf_kernel, imq_kernel, linear_kernel,
    bandwidth_median, bandwidth_scott, bandwidth_silverman,
    bandwidth_fixed, list_bandwidth_rules,
)


__all__ = [
    "SVGD", "SVGDResult",
    "register_kernel", "get_kernel", "list_kernels",
    "rbf_kernel", "imq_kernel", "linear_kernel",
    "bandwidth_median", "bandwidth_scott", "bandwidth_silverman",
    "bandwidth_fixed", "list_bandwidth_rules",
]

__version__ = "0.0.8"
