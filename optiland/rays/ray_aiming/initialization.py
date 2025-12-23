"""Ray Aiming Initialization Module

This module implements the initialization logic for determining the physical
aperture stop size (Stop Radius) before the main Ray Aiming iteration begins.
It uses a Strategy Pattern to handle different ways of calculating this radius.

Kramer Harrison, 2025
"""

from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.rays import RealRays

if TYPE_CHECKING:
    from optiland.optic import Optic


class StopSizeStrategy(abc.ABC):
    """Abstract base class for stop size determination strategies."""

    def __init__(self, optic: Optic):
        self.optic = optic

    @abc.abstractmethod
    def calculate_stop_radius(self) -> float:
        """Calculate the radius of the stop surface."""
        pass


class FloatByStopStrategy(StopSizeStrategy):
    """Strategy for 'Float By Stop Size' aperture type.

    Simply returns the user-defined semi-diameter of the Stop Surface.
    """

    def calculate_stop_radius(self) -> float:
        stop_index = self.optic.surface_group.stop_index
        surface = self.optic.surface_group.surfaces[stop_index]

        # Check for explicit aperture object first
        if surface.aperture and hasattr(surface.aperture, "r_max"):
            return surface.aperture.r_max
        elif surface.aperture and hasattr(surface.aperture, "x_max"):
            return (surface.aperture.x_max + surface.aperture.y_max) / 2.0

        # Fallback to semi_aperture attribute
        return float(surface.semi_aperture)


class ParaxialReferenceStrategy(StopSizeStrategy):
    """Strategy using paraxial ray trace to determine stop radius.

    Traces a Paraxial Marginal Ray from the center of the object to the
    Stop Surface.
    """

    def calculate_stop_radius(self) -> float:
        stop_index = self.optic.surface_group.stop_index
        para = self.optic.paraxial

        # Determine marginal ray height at the stop surface
        y_marginal, _ = para.marginal_ray()
        return be.abs(float(y_marginal[stop_index]))


class RealReferenceStrategy(StopSizeStrategy):
    """Strategy using real ray trace to determine stop radius.

    Traces a Real Ray from the center of the object toward the edge of the
    Paraxial Entrance Pupil. Fallbacks to ParaxialReferenceStrategy on failure.
    """

    def calculate_stop_radius(self) -> float:
        try:
            return self._trace_real_marginal_ray()
        except Exception as e:
            warnings.warn(
                f"RealReferenceStrategy failed: {e}. "
                "Falling back to ParaxialReferenceStrategy.",
                stacklevel=2,
            )
            fallback = ParaxialReferenceStrategy(self.optic)
            return fallback.calculate_stop_radius()

    def _trace_real_marginal_ray(self) -> float:
        wavelength = self.optic.primary_wavelength
        EPL = float(self.optic.paraxial.EPL())
        EPD = float(self.optic.paraxial.EPD())

        stop_index = self.optic.surface_group.stop_index

        # Determine launch ray parameters (x, y, z, L, M, N)
        obj_surf = self.optic.object_surface
        if obj_surf and obj_surf.is_infinite:
            # For infinite objects, we launch a ray parallel to the axis
            # starting slightly before surface 1 to ensure robust intersection.
            z_surf1 = self.optic.surface_group.surfaces[1].geometry.cs.z
            z_start = z_surf1 - 100.0

            y_start = EPD / 2.0
            x_start = 0.0

            rays = RealRays(
                x=be.array([x_start]),
                y=be.array([y_start]),
                z=be.array([z_start]),
                L=be.array([0.0]),
                M=be.array([0.0]),
                N=be.array([1.0]),
                wavelength=be.array([wavelength]),
                intensity=be.array([1.0]),
            )
            start_surf_idx = 1
        else:
            # Finite object: Ray starts at (0, 0, obj_z) pointing to pupil edge
            obj_z = obj_surf.geometry.cs.z

            # Target point: (0, EPD/2, EPL)
            target_y = EPD / 2.0
            target_z = EPL

            dy = target_y - 0.0
            dz = target_z - obj_z
            mag = be.sqrt(dy**2 + dz**2)

            L = 0.0
            M = dy / mag
            N = dz / mag

            rays = RealRays(
                x=be.array([0.0]),
                y=be.array([0.0]),
                z=be.array([obj_z]),
                L=be.array([L]),
                M=be.array([M]),
                N=be.array([N]),
                wavelength=be.array([wavelength]),
                intensity=be.array([1.0]),
            )
            start_surf_idx = 1

        # Trace from start surface up to the stop surface
        for i in range(start_surf_idx, stop_index + 1):
            self.optic.surface_group.surfaces[i].trace(rays)
            if be.any(be.isnan(rays.x)):
                raise ValueError("Ray trace resulted in NaNs (TIR or missed surface).")

        # Return intersection radial height at Stop
        return float(be.sqrt(rays.x[0] ** 2 + rays.y[0] ** 2))


def get_stop_radius_strategy(optic: Optic, aiming_mode: str) -> StopSizeStrategy:
    """Factory function to select the appropriate stop size strategy.

    Args:
        optic: The optical system.
        aiming_mode: The ray aiming mode ('paraxial', 'iterative', 'robust').

    Returns:
        The instantiated strategy instance.
    """
    if optic.aperture and optic.aperture.ap_type == "float_by_stop_size":
        return FloatByStopStrategy(optic)

    if aiming_mode in ["iterative", "robust"]:
        return RealReferenceStrategy(optic)

    return ParaxialReferenceStrategy(optic)
