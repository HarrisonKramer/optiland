"""This package contains the mathematical basis for Forbes polynomials, adapted
for the Optiland backend.
"""

from .geometry import ForbesGeometry
from .qpoly import (
    compute_z_zprime_Q2d,
    Q2d_nm_c_to_a_b,
)

__all__ = [
    "ForbesGeometry",
    "compute_z_zprime_Q2d",
    "Q2d_nm_c_to_a_b",
]