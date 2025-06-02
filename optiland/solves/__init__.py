"""Optiland Solves Subpackage.

Provides classes for applying solves to optical systems. A solve is an
operation that adjusts a property of the optic or a surface to satisfy a
specific condition. For example, a solve can adjust the height of a marginal
ray to a specified value on a specific surface.
"""

from optiland.solves.base import BaseSolve
from optiland.solves.chief_ray_height import ChiefRayHeightSolve
from optiland.solves.factory import SolveFactory
from optiland.solves.marginal_ray_height import MarginalRayHeightSolve
from optiland.solves.quick_focus import QuickFocusSolve
from optiland.solves.solve_manager import SolveManager

__all__ = [
    "BaseSolve",
    "SolveManager",
    "MarginalRayHeightSolve",
    "QuickFocusSolve",
    "ChiefRayHeightSolve",
    "SolveFactory",
]
