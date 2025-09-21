"""
Simple-Stein-SVGD: A minimal Python package for Stein Variational Gradient Descent
"""

from .svgd import SVGD, SVGDResult
from .kernels import rbf_kernel, rbf_kernel_stats

__all__ = ["SVGD", "SVGDResult", "rbf_kernel", "rbf_kernel_stats"]

__version__ = "0.0.5"
