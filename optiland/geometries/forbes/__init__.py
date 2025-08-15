"""This package contains the mathematical basis for Forbes polynomials, adapted
for the Optiland backend.
"""

from __future__ import annotations

from .geometry import ForbesQ2dGeometry, ForbesQbfsGeometry
from .qpoly import (
    Q2d_nm_c_to_a_b,
    compute_z_zprime_Q2d,
)

__all__ = [
    "ForbesQ2dGeometry",
    "ForbesQbfsGeometry",
    "compute_z_zprime_Q2d",
    "Q2d_nm_c_to_a_b",
]
