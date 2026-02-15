"""
Provides a phase profile based on a height map and dispersive material.
"""

from __future__ import annotations

from optiland import backend as be
from optiland.materials.base import BaseMaterial
from optiland.phase.base import BasePhaseProfile

try:
    from scipy.interpolate import RectBivariateSpline
except ImportError:
    RectBivariateSpline = None


class HeightProfile(BasePhaseProfile):
    """A phase profile defined by a height map and a dispersive material.

    The phase is calculated as:
        phi(x, y, λ) = (2π / λ) * (n_material(λ) - 1) * h(x, y)

    Assumes air as the reference medium.

    Args:
        x_coords (be.Array): X-coordinates of the height map grid.
        y_coords (be.Array): Y-coordinates of the height map grid.
        height_map (be.Array): Height values at grid points
            with shape (len(y_coords), len(x_coords)).
        material: Material providing wavelength-dependent refractive index n(λ).
    """

    phase_type = "height_profile"

    def __init__(
        self,
        x_coords: be.Array,
        y_coords: be.Array,
        height_map: be.Array,
        material: BaseMaterial,
    ):
        if RectBivariateSpline is None:
            raise ImportError(
                "scipy is required for HeightProfile. Install with: pip install scipy"
            )

        self.x_coords = be.to_numpy(x_coords)
        self.y_coords = be.to_numpy(y_coords)
        self.height_map = be.to_numpy(height_map)
        self.material = material

        self._spline = RectBivariateSpline(
            self.y_coords,
            self.x_coords,
            self.height_map,
        )

    def _interpolate_height(self, x: be.Array, y: be.Array) -> be.Array:
        return self._spline.ev(be.to_numpy(y), be.to_numpy(x))

    def _interpolate_gradient(
        self, x: be.Array, y: be.Array
    ) -> tuple[be.Array, be.Array]:
        x_np = be.to_numpy(x)
        y_np = be.to_numpy(y)
        dh_dx = self._spline.ev(y_np, x_np, dy=1)
        dh_dy = self._spline.ev(y_np, x_np, dx=1)
        return dh_dx, dh_dy

    def get_phase(
        self,
        x: be.Array,
        y: be.Array,
        wavelength: be.Array,
    ) -> be.Array:
        h = self._interpolate_height(x, y)
        n = self.material.n(wavelength)
        return 2 * be.pi / (wavelength * 1e-3) * (n - 1.0) * h

    def get_gradient(
        self,
        x: be.Array,
        y: be.Array,
        wavelength: be.Array,
    ) -> tuple[be.Array, be.Array, be.Array]:
        dh_dx, dh_dy = self._interpolate_gradient(x, y)
        n = self.material.n(wavelength)
        factor = 2 * be.pi / (wavelength * 1e-3) * (n - 1.0)
        return factor * dh_dx, factor * dh_dy, be.zeros_like(x)

    def get_paraxial_gradient(
        self,
        y: be.Array,
        wavelength: be.Array,
    ) -> be.Array:
        dh_dy = self._spline.ev(
            be.to_numpy(y),
            be.zeros_like(y),
            dx=1,
        )
        n = self.material.n(wavelength)
        return 2 * be.pi / (wavelength * 1e-3) * (n - 1.0) * dh_dy

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["x_coords"] = self.x_coords.tolist()
        data["y_coords"] = self.y_coords.tolist()
        data["height_map"] = self.height_map.tolist()
        data["material"] = getattr(self.material, "name", str(self.material))
        return data

    @classmethod
    def from_dict(cls, data: dict) -> HeightProfile:
        return cls(
            x_coords=be.array(data["x_coords"]),
            y_coords=be.array(data["y_coords"]),
            height_map=be.array(data["height_map"]),
            material=data["material"],
        )