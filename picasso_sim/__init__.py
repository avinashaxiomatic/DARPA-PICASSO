"""
PICASSO Simulation Framework
=============================

A Python framework for simulating and analyzing large-scale photonic
Mach-Zehnder Interferometer (MZI) meshes, with focus on:

- Error accumulation and propagation analysis
- Random matrix theory tools for condition number statistics
- Bayesian inference for noise localization
- Surrogate model training
- Adjoint-based optimization and calibration

Developed in support of DARPA PICASSO program goals.
"""

__version__ = "0.1.0"
__author__ = "PICASSO Team"

from . import core
from . import analysis
from . import random_matrix

__all__ = ["core", "analysis", "random_matrix"]
