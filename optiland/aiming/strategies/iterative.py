from __future__ import annotations
from typing import TYPE_CHECKING
import optiland.backend as be
from scipy.optimize import least_squares
from optiland.aiming.strategies.base import RayAimingStrategy
from optiland.aiming.strategies.paraxial import ParaxialAimingStrategy
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from optiland.backend import ndarray
    from optiland.optic.optic import Optic


class IterativeAimingStrategy(RayAimingStrategy):
    """Iterative ray aiming using scipy.optimize.least_squares.
    """

    def __init__(
        self,
        max_iter: int = 20,
        tol: float = 1e-9,
        robust_fail_penalty: float = 5.0,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
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
            mode = "direction"
            v0 = be.stack([rays0.L, rays0.M], axis=-1)
        else:
            mode = "aimpoint"
            optic.surface_group.trace(rays0)
            stop_idx = optic.surface_group.stop_index
            x_stop = optic.surface_group.x[stop_idx]
            y_stop = optic.surface_group.y[stop_idx]
            v0 = be.stack([x_stop, y_stop], axis=-1)
            if be.any(be.isnan(v0)):
                v0 = be.zeros_like(v0)

        v_optimized = be.copy(v0)
        for i in range(v0.shape[0]):

            def residual_func(v_i):
                v_i_reshaped = be.reshape(v_i, (1, 2))
                single_ray0 = RealRays(
                    be.atleast_1d(rays0.x[i]), be.atleast_1d(rays0.y[i]), be.atleast_1d(rays0.z[i]),
                    be.atleast_1d(rays0.L[i]), be.atleast_1d(rays0.M[i]), be.atleast_1d(rays0.N[i]),
                    be.atleast_1d(rays0.i[i]), be.atleast_1d(rays0.w[i])
                )
                ray_i = self._rays_from_variables(
                    optic, v_i_reshaped, single_ray0,
                    be.atleast_1d(Hx[i]), be.atleast_1d(Hy[i]),
                    be.atleast_1d(Px[i]), be.atleast_1d(Py[i]),
                    wavelength, mode
                )
                optic.surface_group.trace(ray_i)
                stop_idx = optic.surface_group.stop_index
                x, y = optic.surface_group.x[stop_idx], optic.surface_group.y[stop_idx]
                failed = be.isnan(x) | be.isnan(y)
                if be.any(failed):
                    return be.full(2, self.robust_fail_penalty)
                r_stop = self._stop_radius(optic)
                p_stop = be.stack([x / r_stop, y / r_stop], axis=-1)
                residual = p_stop - p_target[i]
                return be.to_numpy(residual.flatten())

            if mode == "direction":
                bounds = (-1.0, 1.0)
            else:
                stop_radius = self._stop_radius(optic)
                bounds = (-2 * stop_radius, 2 * stop_radius)

            res = least_squares(
                residual_func, be.to_numpy(v0[i]), method="trf",
                xtol=self.tol, max_nfev=self.max_iter * 5, bounds=bounds
            )
            v_optimized[i] = be.array(res.x)

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
        self, optic: Optic, v: ndarray, rays0: RealRays, Hx: ndarray, Hy: ndarray,
        Px: ndarray, Py: ndarray, wavelength: float, mode: str
    ) -> RealRays:
        if mode == "direction":
            L, M = v[..., 0], v[..., 1]
            s2 = L * L + M * M
            if be.any(s2 > 1.0):
                scale = be.sqrt(1.0 / s2)
                L, M = L * scale, M * scale
            N = be.sqrt(be.maximum(0.0, 1.0 - (L * L + M * M)))
            return RealRays(rays0.x, rays0.y, rays0.z, L, M, N, rays0.i, rays0.w)

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
