"""Robust Ray Aiming Module

This module implements a recursive robust ray aiming algorithm.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.rays.ray_aiming.iterative import IterativeRayAimer
from optiland.rays.ray_aiming.registry import register_aimer

if TYPE_CHECKING:
    from optiland.optic import Optic


@register_aimer("robust")
class RobustRayAimer(BaseRayAimer):
    """
    This class implements a robust ray aiming algorithm designed to handle
    challenging optical systems where standard iterative methods might fail.
    It employs a recursive subdivision approach to explore the pupil space,
    ensuring that rays are correctly mapped from the object or entrance
    pupil to the stop surface.

    Attributes:
        optic (Optic): The optical system instance.
        max_iter (int): Maximum number of iterations for the internal solver.
        tol (float): Numerical tolerance for convergence.
        scale_fields (bool): If True, scales field coordinates during the
            aiming process to improve numerical stability.
    """

    def __init__(
        self,
        optic: Optic,
        max_iter: int = 20,
        tol: float = 1e-8,
        scale_fields: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the RobustRayAimer.

        Args:
            optic (Optic): The optical system to aim rays for.
            max_iter (int, optional): Maximum number of iterations. Defaults to 20.
            tol (float, optional): Error tolerance for convergence. Defaults to 1e-8.
            scale_fields (bool, optional): If True, scales field coordinates.
                Defaults to True.
            **kwargs: Additional keyword arguments passed to BaseRayAimer.
        """
        super().__init__(optic, **kwargs)
        self.scale_fields = scale_fields
        self._iterative = IterativeRayAimer(optic, max_iter=max_iter, tol=tol)
        self._paraxial = self._iterative._paraxial_aimer

    def aim_rays(
        self,
        fields: tuple,
        wavelengths: Any,
        pupil_coords: tuple,
        initial_guess: tuple | None = None,
    ) -> tuple:
        """Calculate ray starting coordinates using recursive robust expansion.

        This method continually deforms the paraxial solution (t=0) into the
        real solution (t=1).

        Args:
            fields (tuple): Field coordinates.
            wavelengths (Any): Wavelengths in microns.
            pupil_coords (tuple): Normalized pupil coordinates.
            initial_guess (tuple | None, optional): Optional starting guess.
                If provided, the method first attempts to solve directly using
                the iterative solver with this guess. If efficient, this skips
                the robust expansion.

        Returns:
            tuple: Solved ray parameters (x, y, z, L, M, N).
        """
        if initial_guess is not None:
            try:
                return self._iterative.aim_rays(
                    fields, wavelengths, pupil_coords, initial_guess=initial_guess
                )
            except ValueError:
                # Fallback to robust method if guess fails
                pass

        # Anchor at t=0 (Paraxial limit)
        p0 = (pupil_coords[0] * 0.0, pupil_coords[1] * 0.0)
        f0 = (fields[0] * 0.0, fields[1] * 0.0) if self.scale_fields else fields

        sol_0 = self._paraxial.aim_rays(f0, wavelengths, p0)

        # Paraxial and Real are identical at t=0
        return self._solve(0.0, 1.0, sol_0, sol_0, fields, wavelengths, pupil_coords)

    def _solve(self, t0, t1, sol0, par0, fields, wl, pup):
        """Recursively solve the interval [t0, t1].

        Args:
            t0, t1: Start and end scalar parameters.
            sol0: Solution at t0.
            par0: Paraxial solution at t0.
            fields: Target field coordinates.
            wl: Wavelengths.
            pup: Target pupil coordinates.

        Returns:
            tuple: Solution at is t1.
        """
        # Base case: interval too small
        if (t1 - t0) < 1e-3:
            return sol0

        # Target setup
        pt = (pup[0] * t1, pup[1] * t1)
        ft = (fields[0] * t1, fields[1] * t1) if self.scale_fields else fields
        par1 = self._paraxial.aim_rays(ft, wl, pt)

        # Differential Predictor: Guess = Paraxial_New + (Real_Old - Paraxial_Old)
        # Unpack
        x0, y0, z0, L0, M0, N0 = sol0
        px0, py0, pz0, pL0, pM0, pN0 = par0
        px1, py1, pz1, pL1, pM1, pN1 = par1

        xg = px1 + (x0 - px0)
        yg = py1 + (y0 - py0)
        zg = pz1 + (z0 - pz0)
        Lg = pL1 + (L0 - pL0)
        Mg = pM1 + (M0 - pM0)

        # Normalize direction guess
        sq = Lg**2 + Mg**2
        if be.any(sq > 1.0):
            f = be.sqrt(sq)
            Lg, Mg = Lg / f, Mg / f
            sq = Lg**2 + Mg**2  # Update sq after scaling

        Ng = be.sqrt(1.0 - sq)
        Ng = be.where(pN1 >= 0, Ng, -Ng)

        # For infinite objects, direction cosines are fixed by the field angle.
        # We must ignore the differential predictor for L, M, N and use
        # the exact values from the new paraxial trace (par1).
        if getattr(self.optic.object_surface, "is_infinite", False):
            Lg, Mg, Ng = pL1, pM1, pN1

        guess = (xg, yg, zg, Lg, Mg, Ng)

        try:
            return self._iterative.aim_rays(ft, wl, pt, initial_guess=guess)
        except ValueError:
            # Subdivision: t0 -> mid -> t1
            tm = (t0 + t1) / 2.0

            # Solve lower half
            sol_m = self._solve(t0, tm, sol0, par0, fields, wl, pup)

            # Get paraxial at mid for next step
            pm = (pup[0] * tm, pup[1] * tm)
            fm = (fields[0] * tm, fields[1] * tm) if self.scale_fields else fields
            par_m = self._paraxial.aim_rays(fm, wl, pm)

            # Solve upper half
            return self._solve(tm, t1, sol_m, par_m, fields, wl, pup)
