""" abstract base class for ray aiming strategies """
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import optiland.backend as be
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from optiland.optic.optic import Optic


class RayAimingStrategy(ABC):
    """Abstract base class for ray aiming strategies."""

    @abstractmethod
    def aim_ray(self, optic: "Optic", Hx: float, Hy: float, Px: float, Py: float, wavelength: float):
        """given an optic, normalized field and pupil coordinates and wavelength, return the ray"""
        raise NotImplementedError


class ParaxialAimingStrategy(RayAimingStrategy):
    """A ray aiming strategy that uses paraxial optics to aim rays."""

    def aim_ray(self, optic: "Optic", Hx: float, Hy: float, Px: float, Py: float, wavelength: float):
        """given an optic, normalized field and pupil coordinates and wavelength, return the ray"""
        vxf, vyf = optic.fields.get_vig_factor(Hx, Hy)
        vx = 1 - be.array(vxf)
        vy = 1 - be.array(vyf)
        x0, y0, z0 = self._get_ray_origins(optic, Hx, Hy, Px, Py, vx, vy)

        if optic.obj_space_telecentric:
            if optic.field_type == "angle":
                raise ValueError(
                    'Field type cannot be "angle" for telecentric object space.',
                )
            if optic.aperture.ap_type == "EPD":
                raise ValueError(
                    'Aperture type cannot be "EPD" for telecentric object space.',
                )
            if optic.aperture.ap_type == "imageFNO":
                raise ValueError(
                    'Aperture type cannot be "imageFNO" for telecentric object space.',
                )

            sin = optic.aperture.value
            z = be.sqrt(1 - sin**2) / sin + z0
            z1 = be.full_like(Px, z)
            x1 = Px * vx + x0
            y1 = Py * vy + y0
        else:
            EPL = optic.paraxial.EPL()
            EPD = optic.paraxial.EPD()

            x1 = Px * EPD * vx / 2
            y1 = Py * EPD * vy / 2
            z1 = be.full_like(Px, EPL)

        mag = be.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        L = (x1 - x0) / mag
        M = (y1 - y0) / mag
        N = (z1 - z0) / mag

        wavelength = be.ones_like(x0) * wavelength
        intensity = be.ones_like(x0)

        return RealRays(x0, y0, z0, L, M, N, intensity, wavelength)

    def _get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        """Calculate the initial positions for rays originating at the object."""
        obj = optic.object_surface
        max_field = optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy
        if obj.is_infinite:
            if optic.field_type == "object_height":
                raise ValueError(
                    'Field type cannot be "object_height" for an object at infinity.',
                )
            if optic.obj_space_telecentric:
                raise ValueError(
                    "Object space cannot be telecentric for an object at infinity.",
                )
            EPL = optic.paraxial.EPL()
            EPD = optic.paraxial.EPD()

            offset = self._get_starting_z_offset(optic)

            # x, y, z positions of ray starting points
            x = -be.tan(be.radians(field_x)) * (offset + EPL)
            y = -be.tan(be.radians(field_y)) * (offset + EPL)
            z = optic.surface_group.positions[1] - offset

            x0 = Px * EPD / 2 * vx + x
            y0 = Py * EPD / 2 * vy + y
            z0 = be.full_like(Px, z)
        else:
            if optic.field_type == "object_height":
                x0 = be.array(field_x)
                y0 = be.array(field_y)
                z0 = obj.geometry.sag(x0, y0) + obj.geometry.cs.z

            elif optic.field_type == "angle":
                EPL = optic.paraxial.EPL()
                z0 = optic.surface_group.positions[0]
                x0 = -be.tan(be.radians(field_x)) * (EPL - z0)
                y0 = -be.tan(be.radians(field_y)) * (EPL - z0)

            if be.size(x0) == 1:
                x0 = be.full_like(Px, x0)
            if be.size(y0) == 1:
                y0 = be.full_like(Px, y0)
            if be.size(z0) == 1:
                z0 = be.full_like(Px, z0)

        return x0, y0, z0

    def _get_starting_z_offset(self, optic):
        """Calculate the starting ray z-coordinate offset for systems with an
        object at infinity. This is relative to the first surface of the optic.

        This method chooses a starting point that is equivalent to the entrance
        pupil diameter of the optic.

        Returns:
            float: The z-coordinate offset relative to the first surface.

        """
        z = optic.surface_group.positions[1:-1]
        offset = optic.paraxial.EPD()
        if len(z) > 0:
            return offset - be.min(z)
        return offset


class IterativeAimingStrategy(RayAimingStrategy):
    """A ray aiming strategy that uses iterative refinement to aim rays."""

    def __init__(
        self,
        max_iter: int = 10,
        tolerance: float = 1e-9,
        damping: float = 0.8,
        step_size_cap: float = 0.5,
    ):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.damping = damping
        self.step_size_cap = step_size_cap
        self.paraxial_aim = ParaxialAimingStrategy()

    def aim_ray(self, optic: "Optic", Hx, Hy, Px, Py, wavelength: float):
        """given an optic, normalized field and pupil coordinates and wavelength, return the ray"""
        # Ensure inputs are arrays
        Hx = be.atleast_1d(Hx)
        Hy = be.atleast_1d(Hy)
        Px = be.atleast_1d(Px)
        Py = be.atleast_1d(Py)

        # Get initial guess from paraxial aiming
        initial_rays = self.paraxial_aim.aim_ray(optic, Hx, Hy, Px, Py, wavelength)

        if optic.object_surface.is_infinite:
            variables = be.stack([initial_rays.L, initial_rays.M], axis=-1)
        else:
            EPL = optic.paraxial.EPL()
            EPD = optic.paraxial.EPD()
            vxf, vyf = optic.fields.get_vig_factor(Hx, Hy)
            vx = 1 - be.array(vxf)
            vy = 1 - be.array(vyf)
            x1 = Px * EPD * vx / 2
            y1 = Py * EPD * vy / 2
            variables = be.stack([x1, y1], axis=-1)

        target_pupil_coords = be.stack([Px, Py], axis=-1)

        for i in range(self.max_iter):
            current_pupil_coords = self._get_pupil_coords(optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength)
            error = target_pupil_coords - current_pupil_coords

            if be.max(be.abs(error)) < self.tolerance:
                break

            J = self._estimate_jacobian_vectorized(optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength)

            try:
                # Add a small value to the diagonal to prevent singular matrix
                J_reg = J + be.eye(J.shape[1]) * 1e-9
                delta = be.linalg.solve(J_reg, error[..., None])[..., 0]
            except (be.linalg.LinAlgError, RuntimeError): # RuntimeError for torch
                # Fallback to gradient descent if Jacobian is singular
                delta = self.damping * be.matmul(be.transpose(J, (0, 2, 1)), error[..., None])[..., 0]

            step = self.damping * delta
            step_norm = be.linalg.norm(step, axis=-1, keepdims=True)
            # Avoid division by zero
            step_norm = be.where(step_norm == 0, 1.0, step_norm)
            step = be.where(step_norm > self.step_size_cap, step / step_norm * self.step_size_cap, step)

            variables += step

        return self._create_rays_from_variables(optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength)

    def _create_rays_from_variables(self, optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength):
        if optic.object_surface.is_infinite:
            L, M = variables[..., 0], variables[..., 1]
            N = be.sqrt(1 - L**2 - M**2)
            return RealRays(
                initial_rays.x, initial_rays.y, initial_rays.z, L, M, N,
                initial_rays.i, initial_rays.w
            )
        else:
            x1, y1 = variables[..., 0], variables[..., 1]
            vxf, vyf = optic.fields.get_vig_factor(Hx, Hy)
            vx = 1 - be.array(vxf)
            vy = 1 - be.array(vyf)
            x0, y0, z0 = self.paraxial_aim._get_ray_origins(optic, Hx, Hy, Px, Py, vx, vy)
            z1 = be.full_like(x1, optic.paraxial.EPL())
            mag = be.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
            L = (x1 - x0) / mag
            M = (y1 - y0) / mag
            N = (z1 - z0) / mag
            return RealRays(x0, y0, z0, L, M, N, initial_rays.i, initial_rays.w)

    def _get_pupil_coords(self, optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength):
        rays = self._create_rays_from_variables(optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength)
        optic.surface_group.trace(rays)
        stop_idx = optic.surface_group.stop_index

        pupil_x = optic.surface_group.x[stop_idx]
        pupil_y = optic.surface_group.y[stop_idx]

        aperture = optic.surface_group.surfaces[stop_idx].aperture
        if aperture is not None:
            stop_radius = aperture.r_max
        else:
            stop_radius = 1e10 # Assume large radius if no aperture

        return be.stack([pupil_x / stop_radius, pupil_y / stop_radius], axis=-1)

    def _estimate_jacobian_vectorized(self, optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength, epsilon=1e-6):
        num_rays = variables.shape[0]
        num_vars = variables.shape[1]

        epsilon_vec = be.eye(num_vars) * epsilon

        vars_plus = variables[:, None, :] + epsilon_vec
        vars_minus = variables[:, None, :] - epsilon_vec

        vars_perturbed = be.reshape(be.stack([vars_plus, vars_minus], axis=1), (num_rays * 2 * num_vars, num_vars))

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
            optic, vars_perturbed, initial_rays_batch, Hx_batch, Hy_batch, Px_batch, Py_batch, wavelength
        )

        pupil_coords_perturbed = be.reshape(pupil_coords_perturbed, (num_rays, 2, num_vars, 2))
        pupil_coords_plus = pupil_coords_perturbed[:, 0, :, :]
        pupil_coords_minus = pupil_coords_perturbed[:, 1, :, :]

        J_transposed = (pupil_coords_plus - pupil_coords_minus) / (2 * epsilon)
        return be.transpose(J_transposed, axes=(0, 2, 1))


class FallbackAimingStrategy(RayAimingStrategy):
    """A ray aiming strategy that falls back to a secondary strategy if the primary fails."""

    def __init__(
        self,
        primary: Optional[RayAimingStrategy] = None,
        secondary: Optional[RayAimingStrategy] = None,
        pupil_error_threshold: float = 1e-2,
    ):
        self.primary = primary if primary is not None else IterativeAimingStrategy()
        self.secondary = secondary if secondary is not None else ParaxialAimingStrategy()
        self.pupil_error_threshold = pupil_error_threshold

    def aim_ray(self, optic: "Optic", Hx: float, Hy: float, Px: float, Py: float, wavelength: float):
        """given an optic, normalized field and pupil coordinates and wavelength, return the ray"""
        try:
            primary_rays = self.primary.aim_ray(optic, Hx, Hy, Px, Py, wavelength)

            # Trace the rays to check for failure and pupil error
            rays_to_trace = RealRays.from_other(primary_rays)
            optic.surface_group.trace(rays_to_trace)

            if be.any(rays_to_trace.fail):
                return self.secondary.aim_ray(optic, Hx, Hy, Px, Py, wavelength)

            # Check pupil error
            stop_idx = optic.surface_group.stop_index
            pupil_x = optic.surface_group.x[stop_idx]
            pupil_y = optic.surface_group.y[stop_idx]

            aperture = optic.surface_group.surfaces[stop_idx].aperture
            if aperture is None or not hasattr(aperture, "r_max") or aperture.r_max <= 0:
                return primary_rays

            stop_radius = aperture.r_max
            actual_px = pupil_x / stop_radius
            actual_py = pupil_y / stop_radius

            error = be.sqrt((actual_px - be.array(Px)) ** 2 + (actual_py - be.array(Py)) ** 2)

            if be.any(error > self.pupil_error_threshold):
                return self.secondary.aim_ray(optic, Hx, Hy, Px, Py, wavelength)

            return primary_rays
        except Exception:
            return self.secondary.aim_ray(optic, Hx, Hy, Px, Py, wavelength)
