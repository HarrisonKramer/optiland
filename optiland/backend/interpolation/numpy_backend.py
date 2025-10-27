"""Provides a spline interpolator for the numpy backend."""

from __future__ import annotations

from scipy.interpolate import RectBivariateSpline

from optiland import backend as be
from optiland.backend.interpolation.base import SplineInterpolator


class NumpySplineInterpolator(SplineInterpolator):
    """A spline interpolator for the numpy backend."""

    def __init__(
        self,
        x_coords: be.Array,
        y_coords: be.Array,
        grid: be.Array,
    ):
        super().__init__(x_coords, y_coords, grid)
        self._recompute_spline()

    def _recompute_spline(self):
        self._spline = RectBivariateSpline(
            be.to_numpy(self.y_coords),
            be.to_numpy(self.x_coords),
            be.to_numpy(self.grid),
        )

    @property
    def grid(self) -> be.Array:
        """The grid of values to interpolate."""
        return self._grid

    @grid.setter
    def grid(self, new_grid: be.Array):
        self._grid = new_grid
        self._recompute_spline()

    def ev(self, y: be.Array, x: be.Array, dx: int = 0, dy: int = 0) -> be.Array:
        """Evaluates the spline at the given coordinates.

        Args:
            y: The y-coordinates of the points of interest.
            x: The x-coordinates of the points of interest.
            dx: The order of the derivative in the x-direction.
            dy: The order of the derivative in the y-direction.

        Returns:
            The interpolated values at the given coordinates.
        """
        return self._spline.ev(be.to_numpy(y), be.to_numpy(x), dx=dx, dy=dy)
