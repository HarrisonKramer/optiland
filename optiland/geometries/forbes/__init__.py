"""This package contains the mathematical basis for Forbes polynomials, adapted
for the Optiland backend.
"""

from __future__ import annotations

from .geometry import (
    ForbesQ2dGeometry,
    ForbesQbfsGeometry,
    ForbesSolverConfig,
    ForbesSurfaceConfig,
)
from .qpoly import (
    compute_z_zprime_q2d,
    q2d_nm_coeffs_to_ams_bms,
)

__all__ = [
    "ForbesQ2dGeometry",
    "ForbesQbfsGeometry",
    "compute_z_zprime_q2d",
    "q2d_nm_coeffs_to_ams_bms",
    "ForbesSurfaceConfig",
    "ForbesSolverConfig",
]
