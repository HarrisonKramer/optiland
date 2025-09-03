"""Model-Based Ray Aiming Strategy.

This module provides a ray aiming strategy that uses a machine learning model
to predict ray directions.

Kramer Harrison, 2025
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.aiming.strategies.base import RayAimingStrategy
from optiland.aiming.strategies.cached import NumpyEncoder
from optiland.aiming.strategies.iterative import IterativeAimingStrategy
from optiland.aiming.strategies.paraxial import ParaxialAimingStrategy
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from optiland.optic.optic import Optic


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

    Attributes:
        max_cache_size: The maximum number of aiming results to store in the
            cache for each optic.
        refit_frequency: The number of new data points to collect before
            refitting the regression model.
        error_tolerance: The maximum allowable residual pupil error for a
            model prediction to be considered acceptable.
        model_type: The type of regression model to use.
        min_samples_for_fit: The minimum number of cached samples required
            before the first model fitting is attempted.
        iterative_strategy: The iterative strategy to use for refinement.
        paraxial_strategy: The paraxial strategy to use for initial guesses.
        optic_caches: A dictionary to store caches for different optics.
    """

    def __init__(
        self,
        max_cache_size: int = 2048,
        refit_frequency: int = 25,
        error_tolerance: float = 1e-6,
        model_type: str = "polynomial",
        min_samples_for_fit: int = 100,
    ):
        """Initializes the ModelBasedAimingStrategy.

        Args:
            max_cache_size: The maximum number of aiming results to store in the
                cache for each optic.
            refit_frequency: The number of new data points to collect before
                refitting the regression model.
            error_tolerance: The maximum allowable residual pupil error for a
                model prediction to be considered acceptable.
            model_type: The type of regression model to use. Currently, only
                "polynomial" is supported.
            min_samples_for_fit: The minimum number of cached samples required
                before the first model fitting is attempted.
        """
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

    def _get_cache_for_optic(self, optic_hash: str) -> dict:
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
        Hx: ArrayLike,
        Hy: ArrayLike,
        Px: ArrayLike,
        Py: ArrayLike,
        wavelength: float,
    ):
        """Aims a ray using a model-based strategy.

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

    def _aim_ray_scalar(
        self,
        optic: Optic,
        optic_cache: dict,
        Hx: float,
        Hy: float,
        Px: float,
        Py: float,
        wavelength: float,
    ) -> RealRays:
        """Helper for scalar inputs to aim_ray."""
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

    def _predict_ray(
        self,
        optic: Optic,
        model,
        Hx: float,
        Hy: float,
        Px: float,
        Py: float,
        wavelength: float,
    ) -> RealRays | None:
        """Predicts a ray using the model."""
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

    def _check_residual_error(
        self, optic: Optic, rays: RealRays, Px: float, Py: float
    ) -> float:
        """Checks the residual pupil error of a predicted ray."""
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

    def _update_cache_and_model(
        self, optic_cache: dict, cache_key_inputs: tuple, refined_ray: RealRays
    ) -> None:
        """Updates the cache and refits the model if necessary."""
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

    def _fit_model(self, optic_cache: dict) -> None:
        """Fits the regression model."""
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
