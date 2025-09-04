"""Iterative Ray Aiming Strategy.

This module provides a ray aiming strategy that uses iterative refinement to
aim rays.

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
    """A ray aiming strategy that uses iterative refinement to aim rays."""

    def __init__(
        self,
        max_iter: int = 10,
        tolerance: float = 1e-9,
        damping: float = 0.8,
        step_size_cap: float = 0.5,
    ):
        """Initializes the IterativeAimingStrategy.

        Args:
            max_iter: The maximum number of iterations to perform.
            tolerance: The tolerance for the residual pupil error.
            damping: The damping factor for the step size.
            step_size_cap: The maximum step size.
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.damping = damping
        self.step_size_cap = step_size_cap
        self.paraxial_aim = ParaxialAimingStrategy()

    def aim(
        self,
        optic: Optic,
        Hx: ndarray,
        Hy: ndarray,
        Px: ndarray,
        Py: ndarray,
        wavelength: float,
    ):
        """Aims a ray using an iterative refinement method.

        Args:
            optic: The optic to aim the ray for.
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            Px: The normalized x pupil coordinate(s).
            Py: The normalized y pupil coordinate(s).
            wavelength: The wavelength of the ray in microns.

        Returns:
            The aimed ray(s).
        """
        # Ensure inputs are arrays
        Hx = be.atleast_1d(Hx)
        Hy = be.atleast_1d(Hy)
        Px = be.atleast_1d(Px)
        Py = be.atleast_1d(Py)

        # Get initial guess from paraxial aiming
        initial_rays = self.paraxial_aim.aim(optic, Hx, Hy, Px, Py, wavelength)

        if optic.object_surface.is_infinite:
            variables = be.stack([initial_rays.L, initial_rays.M], axis=-1)
        else:
            EPD = optic.paraxial.EPD()
            vxf, vyf = optic.fields.get_vig_factor(Hx, Hy)
            vx = 1 - be.array(vxf)
            vy = 1 - be.array(vyf)
            x1 = Px * EPD * vx / 2
            y1 = Py * EPD * vy / 2
            variables = be.stack([x1, y1], axis=-1)

        target_pupil_coords = be.stack([Px, Py], axis=-1)

        for _i in range(self.max_iter):
            current_pupil_coords = self._get_pupil_coords(
                optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength
            )
            error = target_pupil_coords - current_pupil_coords

            if be.max(be.abs(error)) < self.tolerance:
                break

            J = self._estimate_jacobian_vectorized(
                optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength
            )

            try:
                # Add a small value to the diagonal to prevent singular matrix
                J_reg = J + be.eye(J.shape[1]) * 1e-9
                delta = be.linalg.solve(J_reg, error[..., None])[..., 0]
            except (be.linalg.LinAlgError, RuntimeError):  # RuntimeError for torch
                # Fallback to gradient descent if Jacobian is singular
                delta = (
                    self.damping
                    * be.matmul(be.transpose(J, (0, 2, 1)), error[..., None])[..., 0]
                )

            step = self.damping * delta
            step_norm = be.linalg.norm(step, axis=-1, keepdims=True)
            # Avoid division by zero
            step_norm = be.where(step_norm == 0, 1.0, step_norm)
            step = be.where(
                step_norm > self.step_size_cap,
                step / step_norm * self.step_size_cap,
                step,
            )

            variables += step

        return self._create_rays_from_variables(
            optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength
        )

    def _create_rays_from_variables(
        self,
        optic: Optic,
        variables: ndarray,
        initial_rays: RealRays,
        Hx: ndarray,
        Hy: ndarray,
        Px: ndarray,
        Py: ndarray,
        wavelength: float,
    ) -> RealRays:
        """Creates a RealRays object from the given variables.

        Args:
            optic: The optic to create the rays for.
            variables: The variables to use for creating the rays.
            initial_rays: The initial rays from the paraxial aim.
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            Px: The normalized x pupil coordinate(s).
            Py: The normalized y pupil coordinate(s).
            wavelength: The wavelength of the ray in microns.

        Returns:
            The created RealRays object.
        """
        if optic.object_surface.is_infinite:
            L, M = variables[..., 0], variables[..., 1]
            # clip L and M to prevent nans in N
            norm = be.sqrt(L**2 + M**2)
            cond = norm > 1.0
            L = be.where(cond, L / norm, L)
            M = be.where(cond, M / norm, M)

            N = be.sqrt(1 - L**2 - M**2)
            return RealRays(
                initial_rays.x,
                initial_rays.y,
                initial_rays.z,
                L,
                M,
                N,
                initial_rays.i,
                initial_rays.w,
            )
        else:
            x1, y1 = variables[..., 0], variables[..., 1]
            vxf, vyf = optic.fields.get_vig_factor(Hx, Hy)
            vx = 1 - be.array(vxf)
            vy = 1 - be.array(vyf)
            x0, y0, z0 = self.paraxial_aim._get_ray_origins(
                optic, Hx, Hy, Px, Py, vx, vy
            )
            z1 = be.full_like(x1, optic.paraxial.EPL())
            mag = be.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
            L = (x1 - x0) / mag
            M = (y1 - y0) / mag
            N = (z1 - z0) / mag
            return RealRays(x0, y0, z0, L, M, N, initial_rays.i, initial_rays.w)

    def _get_pupil_coords(
        self,
        optic: Optic,
        variables: ndarray,
        initial_rays: RealRays,
        Hx: ndarray,
        Hy: ndarray,
        Px: ndarray,
        Py: ndarray,
        wavelength: float,
    ) -> ndarray:
        """Gets the pupil coordinates for the given variables.

        Args:
            optic: The optic to get the pupil coordinates for.
            variables: The variables to use for getting the pupil coordinates.
            initial_rays: The initial rays from the paraxial aim.
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            Px: The normalized x pupil coordinate(s).
            Py: The normalized y pupil coordinate(s).
            wavelength: The wavelength of the ray in microns.

        Returns:
            The pupil coordinates.
        """
        rays = self._create_rays_from_variables(
            optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength
        )
        optic.surface_group.trace(rays)
        stop_idx = optic.surface_group.stop_index

        pupil_x = optic.surface_group.x[stop_idx]
        pupil_y = optic.surface_group.y[stop_idx]

        # check for failed rays (nans) and replace with a large number
        # this will push the solver away from regions that cause ray failures
        failed_rays = be.isnan(pupil_x) | be.isnan(pupil_y)
        if be.any(failed_rays):
            pupil_x = be.where(failed_rays, 1e10, pupil_x)
            pupil_y = be.where(failed_rays, 1e10, pupil_y)

        aperture = optic.surface_group.surfaces[stop_idx].aperture
        stop_radius = aperture.r_max if aperture is not None else 1e10

        return be.stack([pupil_x / stop_radius, pupil_y / stop_radius], axis=-1)

    def _estimate_jacobian_vectorized(
        self,
        optic: Optic,
        variables: ndarray,
        initial_rays: RealRays,
        Hx: ndarray,
        Hy: ndarray,
        Px: ndarray,
        Py: ndarray,
        wavelength: float,
        epsilon: float = 1e-6,
    ) -> ndarray:
        """Estimates the Jacobian matrix using vectorized operations.

        Args:
            optic: The optic to estimate the Jacobian for.
            variables: The variables to use for estimating the Jacobian.
            initial_rays: The initial rays from the paraxial aim.
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            Px: The normalized x pupil coordinate(s).
            Py: The normalized y pupil coordinate(s).
            wavelength: The wavelength of the ray in microns.
            epsilon: The epsilon value to use for the finite difference method.

        Returns:
            The estimated Jacobian matrix.
        """
        num_rays = variables.shape[0]
        num_vars = variables.shape[1]

        epsilon_vec = be.eye(num_vars) * epsilon

        vars_plus = variables[:, None, :] + epsilon_vec
        vars_minus = variables[:, None, :] - epsilon_vec

        vars_perturbed = be.reshape(
            be.stack([vars_plus, vars_minus], axis=1),
            (num_rays * 2 * num_vars, num_vars),
        )

        def repeat_param(param, n_repeats):
            if be.size(param) > 1:
                return be.tile(param, (n_repeats, 1))
            else:
                return be.repeat(param, n_repeats * num_rays)

        batch_repeats = num_vars * 2

        # Create a new initial_rays object for the batch
        initial_rays_batch = RealRays(
            repeat_param(initial_rays.x, batch_repeats),
            repeat_param(initial_rays.y, batch_repeats),
            repeat_param(initial_rays.z, batch_repeats),
            repeat_param(initial_rays.L, batch_repeats),
            repeat_param(initial_rays.M, batch_repeats),
            repeat_param(initial_rays.N, batch_repeats),
            repeat_param(initial_rays.i, batch_repeats),
            repeat_param(initial_rays.w, batch_repeats),
        )

        Hx_batch = repeat_param(Hx, batch_repeats)
        Hy_batch = repeat_param(Hy, batch_repeats)
        Px_batch = repeat_param(Px, batch_repeats)
        Py_batch = repeat_param(Py, batch_repeats)

        pupil_coords_perturbed = self._get_pupil_coords(
            optic,
            vars_perturbed,
            initial_rays_batch,
            Hx_batch,
            Hy_batch,
            Px_batch,
            Py_batch,
            wavelength,
        )

        pupil_coords_perturbed = be.reshape(
            pupil_coords_perturbed, (num_rays, 2, num_vars, 2)
        )
        pupil_coords_plus = pupil_coords_perturbed[:, 0, :, :]
        pupil_coords_minus = pupil_coords_perturbed[:, 1, :, :]

        J_transposed = (pupil_coords_plus - pupil_coords_minus) / (2 * epsilon)
        return be.transpose(J_transposed, axes=(0, 2, 1))
