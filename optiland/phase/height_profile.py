"""
Provides a phase profile based on a height map and dispersive material.

Gustavo Vasconcelos, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland import backend as be
from optiland.phase.base import BasePhaseProfile
from optiland.phase.interpolators import GridInterpolator

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial


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
        self.backend = be.get_backend()
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.height_map = height_map
        self.material = material

        self._interp = GridInterpolator(x_coords, y_coords, height_map)

    def _interpolate_height(self, x: be.Array, y: be.Array) -> be.Array:
        return self._interp.height(x, y)

    def _interpolate_gradient(
        self, x: be.Array, y: be.Array
    ) -> tuple[be.Array, be.Array]:
        return self._interp.gradient(x, y)

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
        x0 = be.zeros_like(y)
        _, dh_dy = self._interpolate_gradient(x0, y)
        n = self.material.n(wavelength)
        return 2 * be.pi / (wavelength * 1e-3) * (n - 1.0) * dh_dy

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["x_coords"] = be.to_numpy(self.x_coords).tolist()
        data["y_coords"] = be.to_numpy(self.y_coords).tolist()
        data["height_map"] = be.to_numpy(self.height_map).tolist()
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
