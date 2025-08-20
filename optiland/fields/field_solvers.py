"""Field Solvers Module

This module contains classes for solving object-space field values
(e.g., height or angle) corresponding to a desired image-space height.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.fields.field_modes import AngleFieldMode, ObjectHeightFieldMode
from optiland.raytrace.paraxial_ray_tracer import (
    ParaxialRayTracer as BaseParaxialRayTracer,
)

if TYPE_CHECKING:
    from optiland.optic import Optic


class ParaxialRayTracer(BaseParaxialRayTracer):
    def trace_generic(self, y, u, z, wavelength, reverse=False, skip=0):
        power = (
            self.optic.n(wavelength)[:-1] - self.optic.n(wavelength)[1:]
        ) / self.optic.surface_group.radii[1:]
        power = be.insert(power, 0, 0)
        return super().trace_generic(y, u, z, wavelength, reverse=reverse, skip=skip)


class ParaxialImageHeightSolver:
    """A private helper class to solve for the object field that produces a
    target paraxial image height.

    This solver encapsulates the logic for finding the object-space field
    (angle or height) that corresponds to a desired paraxial image height. It
    uses a linear relationship derived from a paraxial chief ray trace.
    """

    def solve(self, optic: Optic, target_image_height: float) -> float:
        """Finds the object field (angle or height) that corresponds to a
        target paraxial image height.

        Args:
            optic: The optical system.
            target_image_height: The desired paraxial image height.

        Returns:
            The corresponding object-space field value (angle in degrees or
            height in lens units).
        """
        if be.isclose(target_image_height, 0.0):
            return 0.0

        # Determine the field mode based on the object surface state
        if optic.object_surface.is_infinite:
            base_mode = AngleFieldMode()
        else:
            base_mode = ObjectHeightFieldMode()

        # Create a paraxial ray tracer for the optic
        ray_tracer = ParaxialRayTracer(optic)

        # 1. Reverse trace from the stop center to find pupil properties
        stop_index = optic.surface_group.stop_index
        pos = optic.surface_group.positions
        z0_rev_trace = pos[-1] - pos[stop_index]
        wavelength = optic.primary_wavelength
        num_surf = optic.surface_group.num_surfaces
        skip = num_surf - stop_index

        y_rev, u_rev = ray_tracer.trace_generic(
            0.0, 0.1, z0_rev_trace, wavelength, reverse=True, skip=skip
        )

        # 2. Get chief ray start parameters for a unit field
        # Use a unit field of 1.0 (degree or mm)
        unit_object_field = 1.0
        u1_chief_start = base_mode.get_chief_ray_start_params(
            optic, y_rev[-1], u_rev[-1]
        )
        u1_chief_start *= unit_object_field

        # Reverse trace with the unit field chief ray's starting slope
        yn_rev, un_rev = ray_tracer.trace_generic(
            0.0, u1_chief_start, z0_rev_trace, wavelength, reverse=True, skip=skip
        )

        # 3. Forward trace the unit-field chief ray
        z0_fwd_trace = optic.surface_group.positions[1]
        y_fwd, _ = ray_tracer.trace_generic(
            -yn_rev[-1, 0], un_rev[-1, 0], z0_fwd_trace[0], wavelength
        )
        y_image_for_unit_field = y_fwd[-1, 0]

        # 4. Solve for the required field using the ratio
        if be.isclose(y_image_for_unit_field, 0.0):
            raise ValueError("A unit field results in zero image height; cannot solve.")

        solved_field = (
            target_image_height / y_image_for_unit_field
        ) * unit_object_field

        return solved_field


class RealImageHeightSolver:
    def __init__(self):
        raise NotImplementedError("Real image height field mode not yet implemented.")
