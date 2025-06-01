"""Coordinate System Factory Module

This module contains the CoordinateSystemFactory class, which is responsible
for generating coordinate system instances based on input parameters. The class
interfaces with the surface factory to ensure that each surface is assigned the
appropriate coordinate system, including positional and rotational attributes.

Kramer Harrison, 2025
"""

from optiland.coordinate_system import CoordinateSystem


class CoordinateSystemFactory:
    """Factory class for creating CoordinateSystem instances.

    This class abstracts the creation of coordinate systems, ensuring that the
    correct transformations are applied to each surface within an optical system.
    """

    def __init__(self, surface_factory):
        self.surface_factory = surface_factory
        # self.last_thickness is removed as it's no longer needed.

    def create(self, index, surface_group, **kwargs):
        """Creates and returns a CoordinateSystem instance.

        Args:
            index (int): The index of the surface within the optical system.
            surface_group (SurfaceGroup): The group containing all surfaces.
            **kwargs: Additional keyword arguments specifying position and rotation.
                - x (float): X-coordinate (if absolute positioning is used).
                - y (float): Y-coordinate (if absolute positioning is used).
                - z (float): Z-coordinate (if absolute positioning is used).
                - dx (float): X displacement relative to the previous surface.
                - dy (float): Y displacement relative to the previous surface.
                - thickness (float): Thickness of the surface (if relative positioning
                    is used).
                - rx (float): Rotation about the X-axis.
                - ry (float): Rotation about the Y-axis.
                - rz (float): Rotation about the Z-axis.

        Returns:
            CoordinateSystem: The configured coordinate system instance.

        Raises:
            ValueError: If conflicting positional parameters are provided.
        """
        if "z" in kwargs:
            if "thickness" in kwargs:
                raise ValueError('Cannot define both "thickness" and "z".')
            if "dx" in kwargs or "dy" in kwargs:
                raise ValueError(
                    'Cannot define "dx" or "dy" when using absolute "x", "y", "z".',
                )

            x = kwargs.get("x", 0)
            y = kwargs.get("y", 0)
            z_coord = kwargs["z"]
            # Absolute positioning is determined by presence of 'z' in kwargs.
            # No need for self.surface_factory.use_absolute_cs
        else:
            # Relative positioning logic
            dx_rel = kwargs.get("dx", 0)
            dy_rel = kwargs.get("dy", 0)

            # This thickness is stored on the current surface for the *next* one.
            # The actual z positioning uses the *previous* surface's thickness.
            current_surface_thickness_for_next_surface = kwargs.get("thickness", 0.0)

            if index == 0:  # Object surface
                # The "thickness" of the object surface can be thought of as the distance
                # from the origin to the object surface itself, or space before the first optical element.
                # If it's positive, it means the object surface is to the left of the origin.
                z_coord = -current_surface_thickness_for_next_surface
            else: # Subsequent surfaces
                previous_surface = surface_group.surfaces[index - 1]
                thickness_from_previous_to_current = previous_surface.thickness
                z_coord = previous_surface.geometry.cs.z + thickness_from_previous_to_current

            x = dx_rel
            y = dy_rel

        rx = kwargs.get("rx", 0)
        ry = kwargs.get("ry", 0)
        rz = kwargs.get("rz", 0)

        return CoordinateSystem(x=x, y=y, z=z_coord, rx=rx, ry=ry, rz=rz)
