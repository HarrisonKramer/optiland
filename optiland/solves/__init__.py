from __future__ import annotations

from .base import BaseSolve
from .curvature import ChiefRayAngleSolve, CurvatureSolve, MarginalRayAngleSolve
from .factory import SolveFactory
from .quick_focus import QuickFocusSolve
from .solve_manager import SolveManager
from .thickness import ChiefRayHeightSolve, MarginalRayHeightSolve, ThicknessSolve

# Backwards compatibility aliases
RayHeightSolveBase = ThicknessSolve

__all__ = [
    "BaseSolve",
    "ThicknessSolve",
    "MarginalRayHeightSolve",
    "ChiefRayHeightSolve",
    "CurvatureSolve",
    "MarginalRayAngleSolve",
    "ChiefRayAngleSolve",
    "QuickFocusSolve",
    "SolveFactory",
    "SolveManager",
    "RayHeightSolveBase",
]
