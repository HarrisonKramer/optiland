"""Provides a base class for spline interpolation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland import backend as be


class SplineInterpolator(ABC):
    """An abstract base class for spline interpolation."""

    def __init__(
        self,
        x_coords: be.Array,
        y_coords: be.Array,
        grid: be.Array,
    ):
        self.x_coords = x_coords
        self.y_coords = y_coords
        self._grid = grid

    @property
    def grid(self) -> be.Array:
        """The grid of values to interpolate."""
        return self._grid

    @grid.setter
    def grid(self, new_grid: be.Array):
        self._grid = new_grid

    @abstractmethod
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
        raise NotImplementedError
