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
    """Iterative ray aiming strategy using Modified Newton-Raphson.

    This class implements an iterative ray aiming algorithm that solves for the
    initial ray coordinates (x, y) or directions (L, M) required to hit a specific
    target on the stop surface. It uses a Modified Newton-Raphson method with
    a paraxial Jacobian estimate and Broyden rank-1 updates to achieve fast
    super-linear convergence without expensive finite-difference recalculations.

    Attributes:
        optic (Optic): The optical system being traced.
        max_iter (int): Maximum number of iterations allowed.
        tol (float): Convergence tolerance for ray aiming error.
        _paraxial_aimer (ParaxialRayAimer): Helper to generate initial guesses.
    """

    def __init__(
        self,
        optic: Optic,
        max_iter: int = 20,
        tol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        """Initialize the IterativeRayAimer.

        Args:
            optic (Optic): The optical system to aim rays for.
            max_iter (int, optional): Maximum number of iterations. Defaults to 20.
            tol (float, optional): Error tolerance for convergence. Defaults to 1e-8.
            **kwargs: Additional keyword arguments passed to BaseRayAimer.
        """
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
        """Calculate ray starting coordinates using iterative aiming.

        This method solves the inverse ray tracing problem to find the starting
        coordinates (on the object surface) or directions (for finite objects)
        such that the ray passes through the specified pupil coordinates on the
        stop surface.

        Args:
            fields (tuple): Field coordinates (Hy, Hx) or (angle_x, angle_y).
            wavelengths (Any): Wavelengths of the rays in microns.
            pupil_coords (tuple): Normalized pupil coordinates (Px, Py).
            initial_guess (tuple | None, optional): Optional starting guess
                (x, y, z, L, M, N). If None, a paraxial guess is used.

        Returns:
            tuple: A tuple containing the solved ray parameters (x, y, z, L, M, N).

        Raises:
            ValueError: If initial guess produces NaNs or if the solver fails
                to converge within max_iter.
        """
        if initial_guess:
            x, y, z, L, M, N = initial_guess
        else:
            x, y, z, L, M, N = self._paraxial_aimer.aim_rays(
                fields, wavelengths, pupil_coords
            )

        # Ensure arrays
        x = be.as_array_1d(x)
        y = be.as_array_1d(y)
        z = be.as_array_1d(z)
        L = be.as_array_1d(L)
        M = be.as_array_1d(M)
        N = be.as_array_1d(N)

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
        # Ensure proper broadcasting for indexing later
        tx = tx * be.ones_like(x)
        ty = ty * be.ones_like(y)

        tol_sq = self.tol**2

        # Initial trace (all rays)
        rays = self._trace_subset(x, y, z, L, M, N, wavelengths, stop_idx, is_inf)
        ex, ey = rays.x - tx, rays.y - ty

        if be.any(be.isnan(ex)):
            raise ValueError(
                "Initial ray aiming guess produced NaNs. "
                "Consider using the 'robust' method instead."
            )

        num_rays = len(x)
        full_indices = be.arange(num_rays)

        # Precompute Paraxial Jacobian Factor
        wl_mean = (
            be.mean(wavelengths) if hasattr(wavelengths, "__len__") else wavelengths
        )
        J_factor = self._get_paraxial_jacobian(float(wl_mean), stop_idx, is_inf)
        if abs(J_factor) < 1e-12:
            J_factor = 1e-12

        # Initialize Jacobian Estimate (J)
        # J = [[j11, j12], [j21, j22]]
        J11 = be.full(num_rays, J_factor)
        J12 = be.zeros(num_rays)
        J21 = be.zeros(num_rays)
        J22 = be.full(num_rays, J_factor)

        # Store previous state for Broyden
        if is_inf:
            p1_prev = be.copy(x)
            p2_prev = be.copy(y)
        else:
            p1_prev = be.copy(L)
            p2_prev = be.copy(M)
        ex_prev = be.copy(ex)
        ey_prev = be.copy(ey)

        for iter_idx in range(self.max_iter):
            # Check convergence
            error_sq = ex**2 + ey**2
            converged = error_sq < tol_sq

            if be.all(converged):
                return x, y, z, L, M, N

            # Active Set Strategy: only process non-converged rays
            active_mask = ~converged
            idx = full_indices[active_mask]

            # Extract active data
            ex_curr = ex[idx]
            ey_curr = ey[idx]
            # Handle wavelengths (could be scalar or array)
            if hasattr(wavelengths, "__len__"):
                wl_active = wavelengths[idx]
            else:
                wl_active = wavelengths

            # Extract current params
            if is_inf:
                p1_curr = x[idx]
                p2_curr = y[idx]
            else:
                p1_curr = L[idx]
                p2_curr = M[idx]

            # Broyden Update (skip first iter)
            if iter_idx > 0:
                # Get deltas from previous step
                dx = p1_curr - p1_prev[idx]
                dy = p2_curr - p2_prev[idx]

                dEx = ex_curr - ex_prev[idx]
                dEy = ey_curr - ey_prev[idx]

                # Broyden Step
                j11_a = J11[idx]
                j12_a = J12[idx]
                j21_a = J21[idx]
                j22_a = J22[idx]

                # J * s
                Js_x = j11_a * dx + j12_a * dy
                Js_y = j21_a * dx + j22_a * dy

                # Residual = y - J*s
                Rx = dEx - Js_x
                Ry = dEy - Js_y

                # Norm sq
                norm_sq = dx**2 + dy**2
                norm_sq = be.maximum(norm_sq, 1e-20)

                # Update J
                J11[idx] += Rx * dx / norm_sq
                J12[idx] += Rx * dy / norm_sq
                J21[idx] += Ry * dx / norm_sq
                J22[idx] += Ry * dy / norm_sq

            # Calculate Inverse J
            j11_a = J11[idx]
            j12_a = J12[idx]
            j21_a = J21[idx]
            j22_a = J22[idx]

            det = j11_a * j22_a - j12_a * j21_a
            det = be.where(be.abs(det) < 1e-12, 1e-12, det)
            inv_det = 1.0 / det

            inv_j11 = j22_a * inv_det
            inv_j12 = -j12_a * inv_det
            inv_j21 = -j21_a * inv_det
            inv_j22 = j11_a * inv_det

            # Compute Step: - J_inv * E
            step_1 = -(inv_j11 * ex_curr + inv_j12 * ey_curr)
            step_2 = -(inv_j21 * ex_curr + inv_j22 * ey_curr)

            # Store state as 'prev' for next iteration (before updating)
            # Use separate check since p1_prev[idx] is a slice assignment
            if is_inf:
                p1_prev[idx] = x[idx]
                p2_prev[idx] = y[idx]
            else:
                p1_prev[idx] = L[idx]
                p2_prev[idx] = M[idx]
            ex_prev[idx] = ex_curr
            ey_prev[idx] = ey_curr

            # Apply Step
            if is_inf:
                x[idx] += step_1
                y[idx] += step_2

                # Verify Trace
                rays_new = self._trace_subset(
                    x[idx],
                    y[idx],
                    z[idx],
                    L[idx],
                    M[idx],
                    N[idx],
                    wl_active,
                    stop_idx,
                    is_inf,
                )
            else:
                L[idx] += step_1
                M[idx] += step_2

                # Renormalize N
                L_active = L[idx]
                M_active = M[idx]
                sq = L_active**2 + M_active**2
                be.clip(sq, 0, 1.0, out=sq)
                N_active = N[idx]
                N[idx] = be.where(N_active >= 0, be.sqrt(1.0 - sq), -be.sqrt(1.0 - sq))

                # Verify Trace
                rays_new = self._trace_subset(
                    x[idx],
                    y[idx],
                    z[idx],
                    L[idx],
                    M[idx],
                    N[idx],
                    wl_active,
                    stop_idx,
                    is_inf,
                )

            ex_new = rays_new.x - tx[idx]
            ey_new = rays_new.y - ty[idx]

            if be.any(be.isnan(ex_new)):
                raise ValueError("Iterative solver diverged (NaNs).")

            # Update errors
            ex[idx] = ex_new
            ey[idx] = ey_new

        if not be.all((ex**2 + ey**2) < tol_sq):
            raise ValueError("Iterative aimer failed to converge.")

        return x, y, z, L, M, N

    def _get_paraxial_jacobian(
        self, wavelength: float, stop_idx: int, is_inf: bool
    ) -> float:
        """Estimate the Jacobian (magnification) using paraxial trace.

        This method performs a paraxial ray trace to estimate the sensitivity of
        the stop height to changes in the initial ray parameter.

        Args:
            wavelength (float): The wavelength for the trace.
            stop_idx (int): The index of the stop surface.
            is_inf (bool): Whether the object is at infinity.

        Returns:
            float: The estimated Jacobian factor (dy_stop / d_param).
        """
        para = self.optic.paraxial
        if is_inf:
            z_start = para.surfaces.positions[1]
            y, _ = para._trace_generic(1.0, 0.0, z_start, wavelength, skip=1)
            return y[stop_idx]
        else:
            obj_z = self.optic.object_surface.geometry.cs.z
            y, _ = para._trace_generic(0.0, 1.0, obj_z, wavelength)
            return y[stop_idx]

    def _trace_subset(
        self,
        x: Any,
        y: Any,
        z: Any,
        L: Any,
        M: Any,
        N: Any,
        wl: Any,
        stop: int,
        is_inf: bool,
    ) -> RealRays:
        """Trace a subset of rays through the system up to the stop surface.

        Args:
            x, y, z: Ray positions.
            L, M, N: Ray direction cosines.
            wl: Wavelengths.
            stop (int): Index of the stop surface.
            is_inf (bool): Whether the object is at infinity (determines start surface).

        Returns:
            RealRays: The traced rays at the stop surface.
        """
        rays = RealRays(x, y, z, L, M, N, intensity=be.ones_like(x), wavelength=wl)
        start = 1 if is_inf else 0
        for i in range(start, stop + 1):
            self.optic.surface_group.surfaces[i].trace(rays)
        return rays
