"""
Simple-Stein-SVGD: A minimal Python package for Stein Variational Gradient Descent
"""

from .svgd import SVGD
from .kernels import rbf_kernel

__all__ = ["SVGD", "rbf_kernel"]
