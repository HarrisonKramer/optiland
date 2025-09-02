""" abstract base class for ray aiming strategies """
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from optiland.optic.optic import Optic


class RayAimingStrategy(ABC):
    """Abstract base class for ray aiming strategies."""

    @abstractmethod
    def aim_ray(self, optic: "Optic", Hx: float, Hy: float, Px: float, Py: float, wavelength: float):
        """given an optic, normalized field and pupil coordinates and wavelength, return the ray"""
        raise NotImplementedError


class ParaxialAimingStrategy(RayAimingStrategy):
    """A ray aiming strategy that uses paraxial optics to aim rays."""

    def aim_ray(self, optic: "Optic", Hx: float, Hy: float, Px: float, Py: float, wavelength: float):
        """given an optic, normalized field and pupil coordinates and wavelength, return the ray"""
        EPL = optic.paraxial.EPL()
        EPD = optic.paraxial.EPD()

        x1 = Px * EPD / 2
        y1 = Py * EPD / 2

        x0, y0, z0 = self._get_object_position(optic, Hx, Hy, x1, y1, EPL)

        u0x = (x1 - x0) / (EPL - z0)
        u0y = (y1 - y0) / (EPL - z0)

        norm = be.sqrt(1 + u0x**2 + u0y**2)
        L = u0x / norm
        M = u0y / norm
        N = 1 / norm

        x0, y0, z0, L, M, N, wavelength, intensity = be.broadcast_arrays(
            x0, y0, z0, L, M, N, wavelength, be.ones_like(x0)
        )

        return RealRays(x0, y0, z0, L, M, N, intensity, wavelength)

    def _get_object_position(self, optic: "Optic", Hx, Hy, x1, y1, EPL):
        """Calculate the position of the object in the paraxial optical system."""
        obj = optic.object_surface
        # In paraxial calculations, often only one field is defined.
        # We assume the same for x and y if not specified.
        max_field = optic.fields.max_field

        field_x = max_field * Hx
        field_y = max_field * Hy

        if obj.is_infinite:
            if optic.field_type == "object_height":
                raise ValueError(
                    'Field type cannot be "object_height" for an object at infinity.',
                )

            x = -be.tan(be.radians(field_x)) * EPL
            y = -be.tan(be.radians(field_y)) * EPL
            z = optic.surface_group.positions[1]

            x0 = x1 + x
            y0 = y1 + y
            z0 = be.ones_like(y1) * z
        elif optic.field_type == "object_height":
            x = -field_x
            y = -field_y
            z = obj.geometry.cs.z

            x0 = be.ones_like(x1) * x
            y0 = be.ones_like(y1) * y
            z0 = be.ones_like(y1) * z

        elif optic.field_type == "angle":
            x = -be.tan(be.radians(field_x))
            y = -be.tan(be.radians(field_y))
            z = optic.surface_group.positions[0]

            x0 = x1 + x
            y0 = y1 + y
            z0 = be.ones_like(y1) * z
        else:
            # Fallback or error for unsupported field types
            raise ValueError(f"Unsupported field type: {optic.field_type}")

        return x0, y0, z0
