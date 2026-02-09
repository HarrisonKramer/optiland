"""
Provides a phase profile defined on a grid.
"""

from __future__ import annotations

from optiland import backend as be
from optiland.phase.base import BasePhaseProfile

try:
    from scipy.interpolate import RectBivariateSpline
except ImportError:
    RectBivariateSpline = None


class GridPhaseProfile(BasePhaseProfile):
    """A phase profile defined by a grid of phase values.

    This class uses 2D spline interpolation to calculate the phase and its
    gradient at arbitrary points.

    .. note::
        This class requires the `scipy` library to be installed. It is also
        currently only compatible with the NumPy backend.

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
        if be.get_backend() == "torch":
            raise NotImplementedError(
                "GridPhaseProfile is not currently supported for the torch backend."
            )
        if RectBivariateSpline is None:
            raise ImportError(
                "scipy is required for GridPhaseProfile. "
                "Please install it with: pip install scipy"
            )

        self.x_coords = be.to_numpy(x_coords)
        self.y_coords = be.to_numpy(y_coords)
        self.phase_grid = be.to_numpy(phase_grid)

        self._spline = RectBivariateSpline(
            self.y_coords, self.x_coords, self.phase_grid
        )

    def get_phase(self, x: be.Array, y: be.Array) -> be.Array:
        """Calculates the phase added by the profile at coordinates (x, y).

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.

        Returns:
            The phase at each (x, y) coordinate.
        """
        return self._spline.ev(be.to_numpy(y), be.to_numpy(x))

    def get_gradient(
        self, x: be.Array, y: be.Array
    ) -> tuple[be.Array, be.Array, be.Array]:
        """Calculates the gradient of the phase at coordinates (x, y).

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.

        Returns:
            A tuple containing the x, y, and z components of the phase
            gradient (d_phi/dx, d_phi/dy, 0).
        """
        x_np = be.to_numpy(x)
        y_np = be.to_numpy(y)
        d_phi_dx = self._spline.ev(y_np, x_np, dy=1)
        d_phi_dy = self._spline.ev(y_np, x_np, dx=1)
        d_phi_dz = be.zeros_like(x)
        return d_phi_dx, d_phi_dy, d_phi_dz

    def get_paraxial_gradient(self, y: be.Array) -> be.Array:
        """Calculates the paraxial phase gradient at y-coordinate.

        This is the gradient d_phi/dy evaluated at x=0.

        Args:
            y: The y-coordinates of the points of interest.

        Returns:
            The paraxial phase gradient at each y-coordinate.
        """
        return self._spline.ev(be.to_numpy(y), be.zeros_like(y), dx=1)

    def to_dict(self) -> dict:
        """Serializes the phase profile to a dictionary.

        Returns:
            A dictionary representation of the phase profile.
        """
        data = super().to_dict()
        data["x_coords"] = self.x_coords.tolist()
        data["y_coords"] = self.y_coords.tolist()
        data["phase_grid"] = self.phase_grid.tolist()
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
