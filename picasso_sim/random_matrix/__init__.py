"""
Random matrix theory tools for photonic mesh analysis.
"""

from .haar import haar_unitary, sample_haar_phases, is_haar_distributed
from .marchenko_pastur import marchenko_pastur_pdf, fit_marchenko_pastur

__all__ = [
    "haar_unitary", "sample_haar_phases", "is_haar_distributed",
    "marchenko_pastur_pdf", "fit_marchenko_pastur"
]
