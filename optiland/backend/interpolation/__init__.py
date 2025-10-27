"""A backend-agnostic spline interpolation module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.backend import get_backend

if TYPE_CHECKING:
    from optiland import backend as be
    from optiland.backend.interpolation.base import SplineInterpolator


def get_spline_interpolator(
    x_coords: be.Array,
    y_coords: be.Array,
    grid: be.Array,
) -> SplineInterpolator:
    """A factory function for the spline interpolator.

    Args:
        x_coords: The x-coordinates of the grid points.
        y_coords: The y-coordinates of the grid points.
        grid: The values at the grid points.

    Returns:
        An instance of a spline interpolator for the selected backend.
    """
    backend = get_backend()
    if backend == "numpy":
        from optiland.backend.interpolation.numpy_backend import (
            NumpySplineInterpolator,
        )

        return NumpySplineInterpolator(x_coords, y_coords, grid)
    elif backend == "torch":
        from optiland.backend.interpolation.torch_backend import (
            TorchSplineInterpolator,
        )

        return TorchSplineInterpolator(x_coords, y_coords, grid)
    else:
        raise ValueError(f"Backend {backend} is not supported.")
