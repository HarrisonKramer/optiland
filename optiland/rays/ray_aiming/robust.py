"""Robust Ray Aiming Module

This module implements the robust ray aiming algorithm, which uses pupil
expansion (continuation method) to solve for ray aiming coordinates in systems
where standard iterative aiming might fail.

Kramer Harrison, 2025
"""

from __future__ import annotations

from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.rays.ray_aiming.iterative import IterativeRayAimer
from optiland.rays.ray_aiming.registry import register_aimer


@register_aimer("robust")
class RobustRayAimer(BaseRayAimer):
    """Robust ray aiming strategy using pupil expansion.

    This aimer slowly expands the pupil size from a small fraction to the full
    size, using the solution from the previous step as the starting guess for
    the next. This improves convergence for highly aberrated systems.

    Args:
        optic: The optical system to aim rays for.
        fractions: List of pupil fractions to solve for. Defaults to
            [0.1, 0.5, 1.0].
        max_iter: Maximum iterations per step. Defaults to 20.
        tol: Convergence tolerance. Defaults to 1e-8.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        optic,
        fractions: list[float] | None = None,
        max_iter: int = 20,
        tol: float = 1e-8,
        **kwargs,
    ):
        super().__init__(optic, **kwargs)
        self.fractions = fractions if fractions else [0.1, 0.5, 1.0]
        self.max_iter = max_iter
        self.tol = tol
        # Internal iterative aimer
        self._iterative_aimer = IterativeRayAimer(optic, max_iter=max_iter, tol=tol)

    def aim_rays(
        self,
        fields: tuple,
        wavelengths: float,
        pupil_coords: tuple,
    ) -> tuple:
        """Calculate ray starting coordinates using robust pupil expansion."""
        Px, Py = pupil_coords

        # Ensure inputs are arrays for consistent scaling
        # (Though scalars might work if multiplied by scalar fraction)

        current_guess = None

        for fraction in self.fractions:
            # Scale target pupil coordinates
            current_pupil = (Px * fraction, Py * fraction)

            # Solve for current fraction
            # If current_guess is None (first step), IterativeRayAimer will
            # use Paraxial(current_pupil) as guess.
            # If current_guess is provided, it uses that.
            current_guess = self._iterative_aimer.aim_rays(
                fields, wavelengths, current_pupil, initial_guess=current_guess
            )

        return current_guess
