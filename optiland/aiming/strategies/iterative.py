from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.optimize import least_squares

import optiland.backend as be
from optiland.aiming.strategies.base import RayAimingStrategy
from optiland.aiming.strategies.paraxial import ParaxialAimingStrategy
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from optiland.backend import ndarray
    from optiland.optic.optic import Optic


class IterativeAimingStrategy(RayAimingStrategy):
    """Iterative ray aiming using scipy.optimize.least_squares."""

    def __init__(
        self,
        max_iter: int = 20,
        tolerance: float = 1e-9,
        robust_fail_penalty: float = 5.0,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tolerance
        self.robust_fail_penalty = robust_fail_penalty
        self.paraxial = ParaxialAimingStrategy()

    def aim(
        self,
        optic: Optic,
        Hx: ndarray,
        Hy: ndarray,
        Px: ndarray,
        Py: ndarray,
        wavelength: float,
    ) -> RealRays:
        Hx, Hy, Px, Py = map(be.atleast_1d, (Hx, Hy, Px, Py))
        rays0 = self.paraxial.aim(optic, Hx, Hy, Px, Py, wavelength)
        p_target = be.stack([Px, Py], axis=-1)

        if optic.object_surface.is_infinite:
            # For infinite objects, we vary the starting position (x, y) on the
            # first surface, while the direction cosines are fixed for a given field.
            mode = "position"
            v0 = be.stack([rays0.x, rays0.y], axis=-1)
        else:
            # For finite objects, we vary the direction cosines by aiming at
            # different points on the stop surface.
            mode = "aimpoint"
            optic.surface_group.trace(rays0)
            stop_idx = optic.surface_group.stop_index
            x_stop = optic.surface_group.x[stop_idx]
            y_stop = optic.surface_group.y[stop_idx]
            v0 = be.stack([x_stop, y_stop], axis=-1)
            if be.any(be.isnan(v0)):
                v0 = be.zeros_like(v0)

        def residual_func(v):
            v = be.array(v)  # convert numpy array from scipy to backend array
            v_reshaped = be.reshape(v, (-1, 2))
            rays = self._rays_from_variables(
                optic, v_reshaped, rays0, Hx, Hy, Px, Py, wavelength, mode
            )
            optic.surface_group.trace(rays)
            stop_idx = optic.surface_group.stop_index
            x, y = optic.surface_group.x[stop_idx], optic.surface_group.y[stop_idx]

            r_stop = self._stop_radius(optic)
            p_stop = be.stack([x / r_stop, y / r_stop], axis=-1)

            residual = p_stop - p_target

            failed = be.isnan(residual)
            residual = be.where(
                failed, be.full_like(residual, self.robust_fail_penalty), residual
            )

            return be.to_numpy(residual.flatten())

        if mode == "position":
            epd = optic.paraxial.EPD()
            bounds = (-2 * epd, 2 * epd)
            # Clip v0 to be within the bounds, with a small margin
            v0 = be.clip(v0, bounds[0] + 1e-6, bounds[1] - 1e-6)
        else:  # aimpoint
            stop_radius = self._stop_radius(optic)
            bounds = (-2 * stop_radius, 2 * stop_radius)

        bounds = (
            be.full(v0.flatten().shape, bounds[0]),
            be.full(v0.flatten().shape, bounds[1]),
        )

        res = least_squares(
            residual_func,
            be.to_numpy(v0.flatten()),
            method="trf",
            xtol=self.tol,
            max_nfev=self.max_iter * 5,
            bounds=bounds,
        )
        v_optimized = be.reshape(be.array(res.x), (-1, 2))

        return self._rays_from_variables(
            optic, v_optimized, rays0, Hx, Hy, Px, Py, wavelength, mode
        )

    @staticmethod
    def _stop_radius(optic: Optic) -> float:
        idx = optic.surface_group.stop_index
        ap = optic.surface_group.surfaces[idx].aperture
        if ap is not None and getattr(ap, "r_max", None) is not None:
            return float(ap.r_max)
        return float(optic.paraxial.EPD()) / 2.0

    def _rays_from_variables(
        self,
        optic: Optic,
        v: ndarray,
        rays0: RealRays,
        Hx: ndarray,
        Hy: ndarray,
        Px: ndarray,
        Py: ndarray,
        wavelength: float,
        mode: str,
    ) -> RealRays:
        if mode == "position":  # Infinite object case
            x, y = v[..., 0], v[..., 1]
            return RealRays(x, y, rays0.z, rays0.L, rays0.M, rays0.N, rays0.i, rays0.w)

        if mode == "aimpoint":  # Finite object case
            x1, y1 = v[..., 0], v[..., 1]
            stop_idx = optic.surface_group.stop_index
            z1 = be.atleast_1d(optic.surface_group.positions[stop_idx])
            z1 = be.full_like(x1, z1)
            vxf, vyf = optic.fields.get_vig_factor(Hx, Hy)
            vx, vy = 1 - be.array(vxf), 1 - be.array(vyf)
            x0, y0, z0 = self.paraxial._get_ray_origins(optic, Hx, Hy, Px, Py, vx, vy)
            dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
            mag = be.sqrt(dx * dx + dy * dy + dz * dz)
            mag = be.where(mag == 0.0, be.ones_like(mag), mag)
            L, M, N = dx / mag, dy / mag, dz / mag
            return RealRays(x0, y0, z0, L, M, N, rays0.i, rays0.w)

        raise ValueError(f"Unknown mode: {mode}")
