"""Iterative Ray Aiming Module

This module implements the iterative ray aiming algorithm with robust
derivative calculation for wide-angle systems.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.rays import RealRays
from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.rays.ray_aiming.paraxial import ParaxialRayAimer
from optiland.rays.ray_aiming.registry import register_aimer

if TYPE_CHECKING:
    from optiland.optic import Optic


@register_aimer("iterative")
class IterativeRayAimer(BaseRayAimer):
    """Iterative ray aiming strategy using Newton-Raphson."""

    def __init__(
        self,
        optic: Optic,
        max_iter: int = 20,
        tol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(optic, **kwargs)
        self.max_iter = max_iter
        self.tol = tol
        self._paraxial_aimer = ParaxialRayAimer(optic)

    def aim_rays(
        self,
        fields: tuple,
        wavelengths: Any,
        pupil_coords: tuple,
        initial_guess: tuple | None = None,
    ) -> tuple:
        """Calculate ray starting coordinates using iterative aiming."""
        if initial_guess:
            x, y, z, L, M, N = initial_guess
        else:
            x, y, z, L, M, N = self._paraxial_aimer.aim_rays(
                fields, wavelengths, pupil_coords
            )

        Px, Py = pupil_coords
        stop_idx = self.optic.surface_group.stop_index
        stop_surf = self.optic.surface_group.surfaces[stop_idx]
        is_inf = getattr(self.optic.object_surface, "is_infinite", False)

        # Determine target coordinates
        if stop_surf.aperture:
            ap = stop_surf.aperture
            if hasattr(ap, "r_max"):
                rx = ry = ap.r_max
            elif hasattr(ap, "x_max"):
                rx, ry = ap.x_max, ap.y_max
            else:
                rx, ry = ap.extent[1], ap.extent[3]
        else:
            rx = ry = self.optic.paraxial.EPD() / 2.0
            if stop_surf.semi_aperture:
                rx = ry = stop_surf.semi_aperture

        tx, ty = Px * rx, Py * ry
        eps = 1e-4
        tol_sq = self.tol**2

        # Initial trace
        rays = self._trace(x, y, z, L, M, N, wavelengths, stop_idx, is_inf)
        ex, ey = rays.x - tx, rays.y - ty

        if be.any(be.isnan(ex)):
            raise ValueError("Initial guess produced NaNs.")

        for _ in range(self.max_iter):
            converged = (ex**2 + ey**2) < tol_sq
            if be.all(converged):
                return x, y, z, L, M, N

            # Update Step
            try:
                if is_inf:
                    dx, dy = self._step_inf(
                        x, y, z, L, M, N, ex, ey, wavelengths, stop_idx, rays, eps
                    )
                    x, y = (
                        x - be.where(converged, 0, dx),
                        y - be.where(converged, 0, dy),
                    )
                else:
                    dL, dM = self._step_fin(
                        x, y, z, L, M, N, ex, ey, wavelengths, stop_idx, rays, eps
                    )
                    L -= be.where(converged, 0, dL)
                    M -= be.where(converged, 0, dM)

                    # Renormalize N
                    sq = L**2 + M**2
                    be.clip(sq, 0, 1.0, out=sq)
                    N = be.where(N >= 0, be.sqrt(1.0 - sq), -be.sqrt(1.0 - sq))

                # Verification Trace
                rays = self._trace(x, y, z, L, M, N, wavelengths, stop_idx, is_inf)
                ex_new, ey_new = rays.x - tx, rays.y - ty

                if be.any(be.isnan(ex_new)):
                    raise ValueError("Iterative solver diverged (NaNs).")

                ex, ey = ex_new, ey_new

            except ValueError:
                raise

        if not be.all((ex**2 + ey**2) < tol_sq):
            raise ValueError("Iterative aimer failed to converge.")

        return x, y, z, L, M, N

    def _step_inf(self, x, y, z, L, M, N, ex, ey, wl, stop, base, eps):
        """Calculate update for infinite conjugates."""
        rx = self._trace(x + eps, y, z, L, M, N, wl, stop, True)
        ry = self._trace(x, y + eps, z, L, M, N, wl, stop, True)

        dx_x = (rx.x - base.x) / eps
        dy_x = (rx.y - base.y) / eps
        dx_y = (ry.x - base.x) / eps
        dy_y = (ry.y - base.y) / eps

        det = dx_x * dy_y - dx_y * dy_x
        inv = 1.0 / be.where(be.abs(det) < 1e-12, 1e-12, det)

        return (dy_y * ex - dx_y * ey) * inv, (-dy_x * ex + dx_x * ey) * inv

    def _step_fin(self, x, y, z, L, M, N, ex, ey, wl, stop, base, eps):
        """Calculate update for finite conjugates."""
        L1, M1, N1 = self._perturb(L + eps, M, N)
        rL = self._trace(x, y, z, L1, M1, N1, wl, stop, False)

        L2, M2, N2 = self._perturb(L, M + eps, N)
        rM = self._trace(x, y, z, L2, M2, N2, wl, stop, False)

        dx_L = (rL.x - base.x) / eps
        dy_L = (rL.y - base.y) / eps
        dx_M = (rM.x - base.x) / eps
        dy_M = (rM.y - base.y) / eps

        det = dx_L * dy_M - dx_M * dy_L
        inv = 1.0 / be.where(be.abs(det) < 1e-12, 1e-12, det)

        return (dy_M * ex - dx_M * ey) * inv, (-dy_L * ex + dx_L * ey) * inv

    def _trace(self, x, y, z, L, M, N, wl, stop, is_inf):
        rays = RealRays(x, y, z, L, M, N, intensity=be.ones_like(x), wavelength=wl)
        start = 1 if is_inf else 0
        for i in range(start, stop + 1):
            self.optic.surface_group.surfaces[i].trace(rays)
        return rays

    def _perturb(self, L, M, N_orig):
        sq = L**2 + M**2
        be.clip(sq, 0, 1.0, out=sq)
        return L, M, be.where(N_orig >= 0, be.sqrt(1.0 - sq), -be.sqrt(1.0 - sq))
