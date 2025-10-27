"""
Provides a phase profile defined on a grid.
"""

from __future__ import annotations

from optiland import backend as be
from optiland.backend.interpolation import get_spline_interpolator
from optiland.phase.base import BasePhaseProfile


class GridPhaseProfile(BasePhaseProfile):
    """A phase profile defined by a grid of phase values.

    This class uses 2D spline interpolation to calculate the phase and its
    gradient at arbitrary points.

    Args:
        x_coords (be.Array): The x-coordinates of the grid points.
        y_coords (be.Array): The y-coordinates of the grid points.
        phase_grid (be.Array): The phase values at the grid points. The shape
            must be (len(y_coords), len(x_coords)).
    """

    phase_type = "grid"

    def __init__(
        self,
        x_coords: be.Array,
        y_coords: be.Array,
        phase_grid: be.Array,
    ):
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.phase_grid = phase_grid

        self._spline = get_spline_interpolator(
            self.x_coords, self.y_coords, self.phase_grid
        )

    def get_phase(self, x: be.Array, y: be.Array) -> be.Array:
        """Calculates the phase added by the profile at coordinates (x, y).

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.

        Returns:
            The phase at each (x, y) coordinate.
        """
        return self._spline.ev(y, x)

    def get_gradient(self, x: be.Array, y: be.Array) -> tuple[be.Array, be.Array]:
        """Calculates the gradient of the phase at coordinates (x, y).

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.

        Returns:
            A tuple containing the x and y components of the phase gradient
            (d_phi/dx, d_phi/dy).
        """
        d_phi_dx = self._spline.ev(y, x, dy=1)
        d_phi_dy = self._spline.ev(y, x, dx=1)
        return d_phi_dx, d_phi_dy

    def get_paraxial_gradient(self, y: be.Array) -> be.Array:
        """Calculates the paraxial phase gradient at y-coordinate.

        This is the gradient d_phi/dy evaluated at x=0.

        Args:
            y: The y-coordinates of the points of interest.

        Returns:
            The paraxial phase gradient at each y-coordinate.
        """
        return self._spline.ev(y, be.zeros_like(y), dx=1)

    def to_dict(self) -> dict:
        """Serializes the phase profile to a dictionary.

        Returns:
            A dictionary representation of the phase profile.
        """
        data = super().to_dict()
        data["x_coords"] = be.to_numpy(self.x_coords).tolist()
        data["y_coords"] = be.to_numpy(self.y_coords).tolist()
        data["phase_grid"] = be.to_numpy(self.phase_grid).tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> GridPhaseProfile:
        """Deserializes a phase profile from a dictionary.

        Args:
            data: A dictionary representation of a phase profile.

        Returns:
            An instance of a `GridPhaseProfile`.
        """
        return cls(
            x_coords=be.array(data["x_coords"]),
            y_coords=be.array(data["y_coords"]),
            phase_grid=be.array(data["phase_grid"]),
        )
