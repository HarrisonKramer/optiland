"""Cached Ray Aiming Strategy.

This module provides a ray aiming strategy that caches results to avoid
recomputation.

Kramer Harrison, 2025
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.aiming.strategies.base import RayAimingStrategy
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from optiland.optic.optic import Optic


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""

    def default(self, obj):
        """Encode numpy data types.

        Args:
            obj: The object to encode.

        Returns:
            The encoded object.
        """
        if isinstance(obj, np.integer | np.floating | np.bool_):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class CachedAimingStrategy(RayAimingStrategy):
    """A ray aiming strategy that caches results to avoid recomputation."""

    def __init__(self, wrapped_strategy: RayAimingStrategy, max_size: int = 1024):
        """Initializes the CachedAimingStrategy.

        Args:
            wrapped_strategy: The strategy to wrap and cache results from.
            max_size: The maximum number of results to cache.
        """
        self.wrapped_strategy = wrapped_strategy
        self.cache = OrderedDict()
        self.max_size = max_size

    def _get_optic_hash(self, optic: Optic) -> str:
        """Generate a hash for the optic based on its dictionary representation."""
        # The default `str` representation of the dict is not canonical, so we sort keys
        optic_dict = optic.to_dict()
        s = json.dumps(optic_dict, sort_keys=True, cls=NumpyEncoder)
        return hashlib.md5(s.encode()).hexdigest()

    def aim(
        self,
        optic: Optic,
        Hx: ArrayLike,
        Hy: ArrayLike,
        Px: ArrayLike,
        Py: ArrayLike,
        wavelength: float,
    ):
        """Aims a ray, using a cache to avoid recomputation.

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
                    self._aim_scalar(optic, optic_hash, hx, hy, px, py, wavelength)
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
            return self._aim_scalar(optic, optic_hash, Hx, Hy, Px, Py, wavelength)

    def _aim_scalar(
        self,
        optic: Optic,
        optic_hash: str,
        Hx: float,
        Hy: float,
        Px: float,
        Py: float,
        wavelength: float,
    ):
        """Helper for scalar inputs to aim."""
        cache_key = (optic_hash, Hx, Hy, Px, Py, wavelength)

        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            cached_result = self.cache[cache_key]
            # Recreate RealRays from cached data. Assuming intensity is 1.0.
            return RealRays(
                *cached_result, intensity=be.array(1.0), wavelength=be.array(wavelength)
            )

        # Cache miss
        rays = self.wrapped_strategy.aim(
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
