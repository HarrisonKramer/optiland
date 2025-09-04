"""Pupil targeting with Levenberg-Marquardt.

This module provides a robust and efficient iterative solver that finds the
object-space inputs that cause rays to hit requested normalized coordinates at
the aperture stop. It supports both infinite and finite object distances and
only uses the Optiland backend abstraction layer `be`.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.aiming.strategies.base import RayAimingStrategy
from optiland.aiming.strategies.paraxial import ParaxialAimingStrategy
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from optiland.backend import ndarray
    from optiland.optic.optic import Optic


class IterativeAimingStrategy(RayAimingStrategy):
    """Iterative ray aiming using Levenberg-Marquardt.

    The solver treats the problem as a nonlinear least squares system
    r(v) = p_stop(v) - p_target = 0, where p_stop are the normalized stop
    coordinates produced by a choice of variables v, and p_target is the
    requested normalized pupil coordinate (Px, Py).

    Variables v are chosen based on object distance:
      - Infinite object distance: v = [L, M] object-space direction cosines.
      - Finite object distance:  v = [x1, y1] a virtual aim point on the stop
        plane expressed in the physical stop coordinate system.

    We compute updates with an LM step per ray in batch:
      (J^T J + lambda I) delta = J^T r
      v <- project(v - delta)

    The Jacobian J is computed by symmetric, per-parameter finite differences
    with step sizes that adapt to parameter scale. The implementation is fully
    vectorized across rays.

    Parameters
    ----------
    max_iter : int
        Maximum number of outer LM iterations.
    tol : float
        Convergence tolerance on the infinity norm of the residual in normalized
        pupil coordinates.
    lambda_init : float
        Initial LM damping. Increased on rejected steps, decreased on accepted
        steps.
    lambda_up : float
        Multiplier to increase damping when a step is rejected.
    lambda_down : float
        Multiplier to decrease damping when a step is accepted.
    fd_abs : float
        Absolute finite difference step base size.
    fd_rel : float
        Relative finite difference step factor, multiplied by |v|.
    step_cap : float | None
        Optional cap on the 2-norm of a variable update per iteration in the
        variable units. If None, no explicit cap is applied beyond LM.
    robust_fail_penalty : float
        Normalized pupil residual that is injected for rays that fail to reach
        the stop (NaN at stop). Larger values push the solution away from
        failure regions.
    """

    def __init__(
        self,
        max_iter: int = 15,
        tol: float = 1e-8,
        lambda_init: float = 1e-2,
        lambda_up: float = 10.0,
        lambda_down: float = 0.3,
        fd_abs: float = 1e-6,
        fd_rel: float = 1e-3,
        step_cap: float | None = None,
        robust_fail_penalty: float = 5.0,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_init = lambda_init
        self.lambda_up = lambda_up
        self.lambda_down = lambda_down
        self.fd_abs = fd_abs
        self.fd_rel = fd_rel
        self.step_cap = step_cap
        self.robust_fail_penalty = robust_fail_penalty
        self.paraxial = ParaxialAimingStrategy()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def aim(
        self,
        optic: Optic,
        Hx: ndarray,
        Hy: ndarray,
        Px: ndarray,
        Py: ndarray,
        wavelength: float,
    ) -> RealRays:
        """Aim rays so they hit (Px, Py) at the stop.

        All arrays may be scalars or 1D arrays and will be broadcast to a
        common 1D shape.
        """
        Hx, Hy, Px, Py = map(be.atleast_1d, (Hx, Hy, Px, Py))

        # Initial guess from paraxial aiming
        rays0 = self.paraxial.aim(optic, Hx, Hy, Px, Py, wavelength)

        # Target, normalized pupil coordinates at the stop
        p_target = be.stack([Px, Py], axis=-1)  # shape (N, 2)

        # Choose variable parameterization and initial variables
        if optic.object_surface.is_infinite:
            mode = "direction"  # variables are [L, M]
            v = be.stack([rays0.L, rays0.M], axis=-1)
        else:
            mode = "aimpoint"  # variables are [x1, y1] on the stop plane
            stop_radius = self._stop_radius(optic)
            # Reasonable initialization: physical target on the stop plane
            v = be.stack([Px * stop_radius, Py * stop_radius], axis=-1)

        lam = be.array(self.lambda_init)

        # Outer LM iterations
        for _ in range(self.max_iter):
            # Residual at current variables
            p_now, failed = self._pupil_coords(
                optic, v, rays0, Hx, Hy, Px, Py, wavelength, mode
            )
            r = p_now - p_target  # shape (N, 2)

            # Penalize failures with a finite residual, not NaNs
            if be.any(failed):
                pen = be.full_like(r, self.robust_fail_penalty)
                r = be.where(failed[..., None], pen, r)

            # Check convergence
            if be.max(be.abs(r)) <= self.tol:
                break

            # Jacobian by symmetric finite differences (vectorized)
            J = self._jacobian(optic, v, rays0, Hx, Hy, Px, Py, wavelength, mode)
            # Shapes: J: (N, 2, 2), r: (N, 2)

            # LM normal equations in batch
            JT = be.transpose(J, (0, 2, 1))
            JTJ = be.matmul(JT, J)  # (N, 2, 2)
            JTr = be.matmul(JT, r[..., None])[..., 0]  # (N, 2)

            I = be.eye(J.shape[-1])  # (2, 2)  # noqa: E741
            JTJ_damped = JTJ + lam * I  # broadcast over batch

            # Solve for the step, guard solver errors
            try:
                delta = be.linalg.solve(JTJ_damped, JTr)  # (N, 2)
                bad = be.any(be.isnan(delta) | be.isinf(delta), axis=-1)
                if be.any(bad):
                    # Fallback to gradient step for those rays
                    gd = JTr
                    delta = be.where(bad[..., None], gd, delta)
            except Exception:
                # Global fallback: simple gradient step
                delta = JTr

            # Optional trust region on step size
            if self.step_cap is not None:
                nrm = be.linalg.norm(delta, axis=-1, keepdims=True)
                # Avoid division by zero
                nrm = be.where(nrm == 0.0, be.ones_like(nrm), nrm)
                scale = be.minimum(be.ones_like(nrm), self.step_cap / nrm)
                delta = delta * scale

            # Try the step with simple gain based adaptation of lambda
            v_trial = self._project_variables(v - delta, mode)
            p_trial, failed_t = self._pupil_coords(
                optic, v_trial, rays0, Hx, Hy, Px, Py, wavelength, mode
            )
            r_trial = p_trial - p_target
            if be.any(failed_t):
                pen = be.full_like(r_trial, self.robust_fail_penalty)
                r_trial = be.where(failed_t[..., None], pen, r_trial)

            # Accept if residual decreased, else reject and increase damping
            norm_now = be.sum(r * r)
            norm_trial = be.sum(r_trial * r_trial)
            accept = norm_trial < norm_now

            v = be.where(accept, v_trial, v)
            lam = be.where(
                accept,
                be.maximum(lam * self.lambda_down, be.array(1e-9)),
                lam * self.lambda_up,
            )

        # Build final rays from variables
        return self._rays_from_variables(
            optic, v, rays0, Hx, Hy, Px, Py, wavelength, mode
        )

    # ------------------------------------------------------------------
    # Implementation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _stop_radius(optic: Optic) -> float:
        idx = optic.surface_group.stop_index
        ap = optic.surface_group.surfaces[idx].aperture
        if ap is not None and getattr(ap, "r_max", None) is not None:
            return float(ap.r_max)
        # Fallback, approximate by EPD/2 if aperture data is missing
        return float(optic.paraxial.EPD()) / 2.0

    @staticmethod
    def _repeat_rows(x: ndarray, k: int) -> ndarray:
        """Repeat rows of a 1D array x, k times, yielding shape (k*N,)."""
        x = be.atleast_1d(x)
        x2 = x[None, ...]
        reps = (k,) + (1,) * x2.ndim
        tiled = be.tile(x2, reps)
        return be.reshape(tiled, (-1,) + x.shape[1:])

    def _pupil_coords(
        self,
        optic: Optic,
        v: ndarray,  # (N, 2)
        rays0: RealRays,
        Hx: ndarray,
        Hy: ndarray,
        Px: ndarray,
        Py: ndarray,
        wavelength: float,
        mode: str,
    ) -> tuple[ndarray, ndarray]:
        """Trace rays for variables v and return normalized stop coords.

        Returns
        -------
        p_stop : ndarray, shape (N, 2)
            Normalized coordinates at the stop, x and y divided by the physical
            stop radius.
        failed : ndarray, shape (N,)
            Boolean mask for rays that failed to reach the stop.
        """
        rays = self._rays_from_variables(
            optic, v, rays0, Hx, Hy, Px, Py, wavelength, mode
        )
        optic.surface_group.trace(rays)
        idx = optic.surface_group.stop_index
        x = optic.surface_group.x[idx]
        y = optic.surface_group.y[idx]

        failed = be.isnan(x) | be.isnan(y)

        r_stop = self._stop_radius(optic)
        p = be.stack([x / r_stop, y / r_stop], axis=-1)
        return p, failed

    def _rays_from_variables(
        self,
        optic: Optic,
        v: ndarray,  # (N, 2)
        rays0: RealRays,
        Hx: ndarray,
        Hy: ndarray,
        Px: ndarray,
        Py: ndarray,
        wavelength: float,
        mode: str,
    ) -> RealRays:
        """Create rays from variables for the selected mode."""
        if mode == "direction":
            L = v[..., 0]
            M = v[..., 1]
            # Keep direction within the unit disk
            s2 = L * L + M * M
            too_big = s2 > 1.0
            # Project onto circle of radius 0.999 if needed
            scale = be.where(too_big, 0.999 / be.sqrt(s2), be.ones_like(s2))
            L = L * scale
            M = M * scale
            N = be.sqrt(be.maximum(0.0, 1.0 - (L * L + M * M)))
            return RealRays(rays0.x, rays0.y, rays0.z, L, M, N, rays0.i, rays0.w)

        # Finite object distance, aim toward a virtual point (x1, y1, z_stop)
        x1 = v[..., 0]
        y1 = v[..., 1]
        z1 = be.full_like(x1, optic.paraxial.EPL())

        # Ray origins from paraxial aim
        vxf, vyf = optic.fields.get_vig_factor(Hx, Hy)
        vx = 1 - be.array(vxf)
        vy = 1 - be.array(vyf)
        x0, y0, z0 = self.paraxial._get_ray_origins(optic, Hx, Hy, Px, Py, vx, vy)

        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        mag = be.sqrt(dx * dx + dy * dy + dz * dz)
        # Protect against a zero length direction
        mag = be.where(mag == 0.0, be.ones_like(mag), mag)
        L = dx / mag
        M = dy / mag
        N = dz / mag
        return RealRays(x0, y0, z0, L, M, N, rays0.i, rays0.w)

    def _jacobian(
        self,
        optic: Optic,
        v: ndarray,  # (N, 2)
        rays0: RealRays,
        Hx: ndarray,
        Hy: ndarray,
        Px: ndarray,
        Py: ndarray,
        wavelength: float,
        mode: str,
    ) -> ndarray:
        """Vectorized symmetric finite difference Jacobian.

        Returns J with shape (N, 2, 2) where J[i] maps dv -> dp_stop.
        """
        N = v.shape[0]
        P = v.shape[1]
        assert P == 2, "Only 2D variable sets are supported"

        # Per parameter step sizes
        abs_step = be.full_like(v, self.fd_abs)
        rel_step = self.fd_rel * be.abs(v)
        h = abs_step + rel_step  # shape (N, 2)

        # Build 2P perturbed stacks: v_plus and v_minus for each parameter
        # For P=2, order is [+e1, -e1, +e2, -e2]
        e = be.eye(P)  # (2, 2)
        E = be.reshape(e, (1, P, P))  # (1, 2, 2)
        H = h[:, None, :]  # (N, 1, 2)
        V = v[:, None, :]  # (N, 1, 2)

        v_plus = V + H * E  # (N, 2, 2)
        v_minus = V - H * E  # (N, 2, 2)

        # Stack into a single batch and reshape to (2P*N, 2)
        vm = be.stack([v_plus, v_minus], axis=1)  # (N, 2, 2, 2)
        vm = be.reshape(vm, (N * 2 * P, P))

        # Repeat the per ray metadata to match the perturbed batch size
        reps = 2 * P

        def rep(x: ndarray) -> ndarray:
            return self._repeat_rows(x, reps)

        rays0_rep = RealRays(
            rep(rays0.x),
            rep(rays0.y),
            rep(rays0.z),
            rep(rays0.L),
            rep(rays0.M),
            rep(rays0.N),
            rep(rays0.i),
            rep(rays0.w),
        )
        Hx_r, Hy_r, Px_r, Py_r = map(rep, (Hx, Hy, Px, Py))

        p_batch, _ = self._pupil_coords(
            optic, vm, rays0_rep, Hx_r, Hy_r, Px_r, Py_r, wavelength, mode
        )  # shape (2P*N, 2)

        # Unpack back to (N, 2, P, 2) then central difference along the 2 axis
        p_batch = be.reshape(p_batch, (N, 2, P, 2))
        p_plus = p_batch[:, 0, :, :]  # (N, P, 2)
        p_minus = p_batch[:, 1, :, :]  # (N, P, 2)

        J_tp = (p_plus - p_minus) / (2.0 * h[:, None, :])  # (N, P, 2)
        # Transpose to (N, 2, P)
        return be.transpose(J_tp, (0, 2, 1))

    @staticmethod
    def _project_variables(v: ndarray, mode: str) -> ndarray:
        """Project variables into a safe set to avoid NaNs.

        - For directions, ensure L^2 + M^2 <= 0.999^2.
        - For aim points, lightly clip to a generous radius envelope per ray.
        """
        if mode == "direction":
            L = v[..., 0]
            M = v[..., 1]
            s2 = L * L + M * M
            too_big = s2 > 0.999 * 0.999
            scale = be.where(too_big, 0.999 / be.sqrt(s2), be.ones_like(s2))
            L = L * scale
            M = M * scale
            return be.stack([L, M], axis=-1)

        # Aim points, trust region style clipping based on current radius
        x1 = v[..., 0]
        y1 = v[..., 1]
        r = be.sqrt(x1 * x1 + y1 * y1)
        # Allow up to 1.5x of current radius to avoid exploding steps
        r_safe = 1.5 * be.maximum(r, be.array(1.0))
        r = be.where(r > r_safe, r_safe, r)
        scale = be.where(
            r == 0.0, be.ones_like(r), r_safe / be.maximum(r, be.array(1e-12))
        )
        x1 = x1 * scale
        y1 = y1 * scale
        return be.stack([x1, y1], axis=-1)
