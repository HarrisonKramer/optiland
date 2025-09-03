"""abstract base class for ray aiming strategies"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from optiland.optic.optic import Optic


class RayAimingStrategy(ABC):
    """Abstract base class for ray aiming strategies."""

    @abstractmethod
    def aim_ray(
        self,
        optic: Optic,
        Hx: float,
        Hy: float,
        Px: float,
        Py: float,
        wavelength: float,
    ):
        """Given an optic, normalized field and pupil coordinates and wavelength,
        return the ray
        """
        raise NotImplementedError


class ParaxialAimingStrategy(RayAimingStrategy):
    """A ray aiming strategy that uses paraxial optics to aim rays."""

    def aim_ray(
        self,
        optic: Optic,
        Hx: float,
        Hy: float,
        Px: float,
        Py: float,
        wavelength: float,
    ):
        """Given an optic, normalized field and pupil coordinates and wavelength,
        return the ray
        """
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

    def aim_ray(self, optic: Optic, Hx, Hy, Px, Py, wavelength: float):
        """Given an optic, normalized field and pupil coordinates and wavelength,
        return the ray
        """
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
        self, optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength
    ):
        if optic.object_surface.is_infinite:
            L, M = variables[..., 0], variables[..., 1]
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
        self, optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength
    ):
        rays = self._create_rays_from_variables(
            optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength
        )
        optic.surface_group.trace(rays)
        stop_idx = optic.surface_group.stop_index

        pupil_x = optic.surface_group.x[stop_idx]
        pupil_y = optic.surface_group.y[stop_idx]

        aperture = optic.surface_group.surfaces[stop_idx].aperture
        stop_radius = aperture.r_max if aperture is not None else 1e10

        return be.stack([pupil_x / stop_radius, pupil_y / stop_radius], axis=-1)

    def _estimate_jacobian_vectorized(
        self, optic, variables, initial_rays, Hx, Hy, Px, Py, wavelength, epsilon=1e-6
    ):
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


class FallbackAimingStrategy(RayAimingStrategy):
    """A ray aiming strategy that falls back to a secondary strategy if the
    primary fails.
    """

    def __init__(
        self,
        primary: RayAimingStrategy | None = None,
        secondary: RayAimingStrategy | None = None,
        pupil_error_threshold: float = 1e-2,
    ):
        self.primary = primary if primary is not None else IterativeAimingStrategy()
        self.secondary = (
            secondary if secondary is not None else ParaxialAimingStrategy()
        )
        self.pupil_error_threshold = pupil_error_threshold

    def aim_ray(
        self,
        optic: Optic,
        Hx: float,
        Hy: float,
        Px: float,
        Py: float,
        wavelength: float,
    ):
        """Given an optic, normalized field and pupil coordinates and wavelength,
        return the ray
        """
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
            if not (aperture and hasattr(aperture, "r_max") and aperture.r_max > 0):
                return primary_rays

            stop_radius = aperture.r_max
            actual_px = pupil_x / stop_radius
            actual_py = pupil_y / stop_radius

            error = be.sqrt(
                (actual_px - be.array(Px)) ** 2 + (actual_py - be.array(Py)) ** 2
            )

            if be.any(error > self.pupil_error_threshold):
                return self.secondary.aim_ray(optic, Hx, Hy, Px, Py, wavelength)

            return primary_rays
        except Exception:
            return self.secondary.aim_ray(optic, Hx, Hy, Px, Py, wavelength)


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(obj, np.integer | np.floating | np.bool_):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class CachedAimingStrategy(RayAimingStrategy):
    """A ray aiming strategy that caches results to avoid recomputation."""

    def __init__(self, wrapped_strategy: RayAimingStrategy, max_size: int = 1024):
        self.wrapped_strategy = wrapped_strategy
        self.cache = OrderedDict()
        self.max_size = max_size

    def _get_optic_hash(self, optic: Optic) -> str:
        """Generate a hash for the optic based on its dictionary representation."""
        # The default `str` representation of the dict is not canonical, so we sort keys
        optic_dict = optic.to_dict()
        s = json.dumps(optic_dict, sort_keys=True, cls=NumpyEncoder)
        return hashlib.md5(s.encode()).hexdigest()

    def aim_ray(
        self,
        optic: Optic,
        Hx: float,
        Hy: float,
        Px: float,
        Py: float,
        wavelength: float,
    ):
        """Given an optic, normalized field and pupil coordinates and wavelength,
        return the ray
        """
        optic_hash = self._get_optic_hash(optic)

        # Handle vectorized inputs by iterating and calling the scalar implementation
        is_vectorized = any(be.size(arg) > 1 for arg in [Hx, Hy, Px, Py])
        if is_vectorized:
            Hx_arr, Hy_arr, Px_arr, Py_arr = map(be.atleast_1d, [Hx, Hy, Px, Py])

            # Find the broadcast size
            size = max(
                len(arr) if hasattr(arr, "__len__") and len(arr) > 1 else 1
                for arr in [Hx_arr, Hy_arr, Px_arr, Py_arr]
            )

            results = []
            for i in range(size):
                hx = Hx_arr[i] if len(Hx_arr) > 1 else Hx_arr[0]
                hy = Hy_arr[i] if len(Hy_arr) > 1 else Hy_arr[0]
                px = Px_arr[i] if len(Px_arr) > 1 else Px_arr[0]
                py = Py_arr[i] if len(Py_arr) > 1 else Py_arr[0]
                results.append(
                    self._aim_ray_scalar(optic, optic_hash, hx, hy, px, py, wavelength)
                )

            # Stack results into a single RealRays object
            return RealRays(
                x=be.stack([r.x for r in results]),
                y=be.stack([r.y for r in results]),
                z=be.stack([r.z for r in results]),
                L=be.stack([r.L for r in results]),
                M=be.stack([r.M for r in results]),
                N=be.stack([r.N for r in results]),
                intensity=be.stack([r.i for r in results]),
                wavelength=be.stack([r.w for r in results]),
            )
        else:
            return self._aim_ray_scalar(optic, optic_hash, Hx, Hy, Px, Py, wavelength)

    def _aim_ray_scalar(
        self,
        optic: Optic,
        optic_hash: str,
        Hx: float,
        Hy: float,
        Px: float,
        Py: float,
        wavelength: float,
    ):
        """Helper for scalar inputs to aim_ray."""
        cache_key = (optic_hash, Hx, Hy, Px, Py, wavelength)

        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            cached_result = self.cache[cache_key]
            # Recreate RealRays from cached data. Assuming intensity is 1.0.
            return RealRays(
                *cached_result, intensity=be.array(1.0), wavelength=be.array(wavelength)
            )

        # Cache miss
        rays = self.wrapped_strategy.aim_ray(
            optic=optic, Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wavelength
        )

        # Ensure result from wrapped strategy is suitable for caching (scalar-like)
        # The wrapped strategy should return a RealRays object where each attribute
        # is a 1-element array
        if be.size(rays.x) > 1:
            # This indicates the wrapped strategy might have returned a vectorized
            # result unexpectedly.
            # The current caching implementation expects to cache individual rays.
            # For simplicity, we'll cache the first ray's data. A more robust
            # solution might be needed.
            result_to_cache = (
                be.to_numpy(rays.x)[0],
                be.to_numpy(rays.y)[0],
                be.to_numpy(rays.z)[0],
                be.to_numpy(rays.L)[0],
                be.to_numpy(rays.M)[0],
                be.to_numpy(rays.N)[0],
            )
        else:
            result_to_cache = (
                be.to_numpy(rays.x).item(),
                be.to_numpy(rays.y).item(),
                be.to_numpy(rays.z).item(),
                be.to_numpy(rays.L).item(),
                be.to_numpy(rays.M).item(),
                be.to_numpy(rays.N).item(),
            )

        self.cache[cache_key] = result_to_cache
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        return rays


class ModelBasedAimingStrategy(RayAimingStrategy):
    """
    A ray aiming strategy that uses a machine learning model to predict ray directions,
    improving performance by caching and learning from past results.

    This strategy maintains a cache of aiming results for each optical system
    configuration. When a new ray aiming request is received, it first attempts to
    predict the initial ray direction using a regression model trained on the cached
    data. If the model's prediction is accurate enough (within a specified
    tolerance), the ray is returned immediately. Otherwise, it falls back to a more
    robust iterative refinement method, and the new, accurate result is used to
    update the cache and incrementally improve the model.

    This approach is designed to accelerate ray aiming in scenarios involving repeated
    calculations on the same or similar optical systems, such as during optimization
    or tolerancing analysis.

    Parameters
    ----------
    max_cache_size : int, optional
        The maximum number of aiming results to store in the cache for each optic,
        by default 2048.
    refit_frequency : int, optional
        The number of new data points to collect before refitting the regression model,
        by default 25.
    error_tolerance : float, optional
        The maximum allowable residual pupil error for a model prediction to be
        considered acceptable, by default 1e-6.
    model_type : str, optional
        The type of regression model to use. Currently, only "polynomial" is
        supported, by default "polynomial".
    min_samples_for_fit : int, optional
        The minimum number of cached samples required before the first model
        fitting is attempted, by default 100.
    """

    def __init__(
        self,
        max_cache_size: int = 2048,
        refit_frequency: int = 25,
        error_tolerance: float = 1e-6,
        model_type: str = "polynomial",
        min_samples_for_fit: int = 100,
    ):
        self.max_cache_size = max_cache_size
        self.refit_frequency = refit_frequency
        self.error_tolerance = error_tolerance
        self.model_type = model_type
        self.min_samples_for_fit = min_samples_for_fit

        self.iterative_strategy = IterativeAimingStrategy()
        self.paraxial_strategy = ParaxialAimingStrategy()
        self.optic_caches = {}  # Stores caches for different optics

    def _get_optic_hash(self, optic: Optic) -> str:
        """Generate a hash for the optic based on its dictionary representation."""
        optic_dict = optic.to_dict()
        s = json.dumps(optic_dict, sort_keys=True, cls=NumpyEncoder)
        return hashlib.md5(s.encode()).hexdigest()

    def _get_cache_for_optic(self, optic_hash: str):
        """Get or create the cache for a given optic hash."""
        if optic_hash not in self.optic_caches:
            self.optic_caches[optic_hash] = {
                "cache": OrderedDict(),
                "model": None,
                "new_points_count": 0,
            }
        return self.optic_caches[optic_hash]

    def aim_ray(
        self,
        optic: Optic,
        Hx: float,
        Hy: float,
        Px: float,
        Py: float,
        wavelength: float,
    ):
        """Given an optic, normalized field and pupil coordinates and wavelength,
        return the ray
        """
        optic_hash = self._get_optic_hash(optic)
        optic_cache = self._get_cache_for_optic(optic_hash)

        is_vectorized = any(be.size(arg) > 1 for arg in [Hx, Hy, Px, Py])
        if is_vectorized:
            Hx_arr, Hy_arr, Px_arr, Py_arr = map(be.atleast_1d, [Hx, Hy, Px, Py])
            size = max(
                len(arr) if hasattr(arr, "__len__") and len(arr) > 1 else 1
                for arr in [Hx_arr, Hy_arr, Px_arr, Py_arr]
            )

            rays_list = []
            for i in range(size):
                hx = Hx_arr[i] if len(Hx_arr) > 1 else Hx_arr[0]
                hy = Hy_arr[i] if len(Hy_arr) > 1 else Hy_arr[0]
                px = Px_arr[i] if len(Px_arr) > 1 else Px_arr[0]
                py = Py_arr[i] if len(Py_arr) > 1 else Py_arr[0]
                rays_list.append(
                    self._aim_ray_scalar(optic, optic_cache, hx, hy, px, py, wavelength)
                )

            return RealRays(
                x=be.stack([r.x for r in rays_list]),
                y=be.stack([r.y for r in rays_list]),
                z=be.stack([r.z for r in rays_list]),
                L=be.stack([r.L for r in rays_list]),
                M=be.stack([r.M for r in rays_list]),
                N=be.stack([r.N for r in rays_list]),
                intensity=be.stack([r.i for r in rays_list]),
                wavelength=be.stack([r.w for r in rays_list]),
            )

        return self._aim_ray_scalar(optic, optic_cache, Hx, Hy, Px, Py, wavelength)

    def _aim_ray_scalar(self, optic, optic_cache, Hx, Hy, Px, Py, wavelength):
        if optic_cache["model"] is not None:
            predicted_ray = self._predict_ray(
                optic, optic_cache["model"], Hx, Hy, Px, Py, wavelength
            )
            if predicted_ray:
                error = self._check_residual_error(optic, predicted_ray, Px, Py)
                if error < self.error_tolerance:
                    return predicted_ray

        refined_ray = self.iterative_strategy.aim_ray(optic, Hx, Hy, Px, Py, wavelength)
        self._update_cache_and_model(
            optic_cache, (Hx, Hy, Px, Py, wavelength), refined_ray
        )
        return refined_ray

    def _predict_ray(self, optic, model, Hx, Hy, Px, Py, wavelength):
        inputs = be.array([[Hx, Hy, Px, Py]])
        try:
            predicted_lm = model.predict(inputs)
            L, M = predicted_lm[0, 0], predicted_lm[0, 1]

            if L**2 + M**2 >= 1.0:
                return None

            N = be.sqrt(1 - L**2 - M**2)
            paraxial_ray = self.paraxial_strategy.aim_ray(
                optic, Hx, Hy, Px, Py, wavelength
            )
            predicted_ray = RealRays(
                paraxial_ray.x,
                paraxial_ray.y,
                paraxial_ray.z,
                be.array(L),
                be.array(M),
                be.array(N),
                intensity=be.ones_like(paraxial_ray.x),
                wavelength=be.array(wavelength),
            )
            predicted_ray.fail = be.zeros_like(predicted_ray.x, dtype=bool)
            return predicted_ray
        except Exception:
            return None

    def _check_residual_error(self, optic, rays, Px, Py):
        rays_to_trace = RealRays.from_other(rays)
        optic.surface_group.trace(rays_to_trace)

        if be.any(rays_to_trace.fail):
            return float("inf")

        stop_idx = optic.surface_group.stop_index
        pupil_x = optic.surface_group.x[stop_idx]
        pupil_y = optic.surface_group.y[stop_idx]

        aperture = optic.surface_group.surfaces[stop_idx].aperture
        if not (aperture and hasattr(aperture, "r_max") and aperture.r_max > 0):
            return 0.0

        stop_radius = aperture.r_max
        actual_px = pupil_x / stop_radius
        actual_py = pupil_y / stop_radius
        error = be.sqrt(
            (actual_px - be.array(Px)) ** 2 + (actual_py - be.array(Py)) ** 2
        )
        return be.to_numpy(error).item()

    def _update_cache_and_model(self, optic_cache, cache_key_inputs, refined_ray):
        cache_key = cache_key_inputs
        if cache_key not in optic_cache["cache"]:
            result_to_cache = (
                be.to_numpy(refined_ray.L).item(),
                be.to_numpy(refined_ray.M).item(),
                be.to_numpy(refined_ray.N).item(),
            )
            optic_cache["cache"][cache_key] = result_to_cache
            optic_cache["new_points_count"] += 1

            if len(optic_cache["cache"]) > self.max_cache_size:
                optic_cache["cache"].popitem(last=False)

            if (
                optic_cache["new_points_count"] >= self.refit_frequency
                and len(optic_cache["cache"]) >= self.min_samples_for_fit
            ):
                self._fit_model(optic_cache)
                optic_cache["new_points_count"] = 0

    def _fit_model(self, optic_cache):
        try:
            from optiland.ml.regression import PolynomialRegression
        except ImportError:
            # This is a fallback for cases where the ml module might not be available.
            # In normal operation, this import should succeed.
            return

        if self.model_type == "polynomial":
            optic_cache["model"] = PolynomialRegression(degree=3)
        else:
            raise NotImplementedError(
                f"Model type '{self.model_type}' is not supported."
            )

        cache_items = list(optic_cache["cache"].items())
        X = be.array(
            [[item[0][0], item[0][1], item[0][2], item[0][3]] for item in cache_items]
        )
        y = be.array([[item[1][0], item[1][1]] for item in cache_items])
        optic_cache["model"].fit(X, y)
