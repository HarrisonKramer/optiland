"""Coordinate System Factory Module

This module contains the CoordinateSystemFactory class, which is responsible
for generating coordinate system instances based on input parameters.

Kramer Harrison, 2025
"""

from __future__ import annotations

from optiland.coordinate_system import CoordinateSystem


class CoordinateSystemFactory:
    """A stateless factory for creating CoordinateSystem instances."""

    @staticmethod
    def create(
        x: float, y: float, z: float, rx: float, ry: float, rz: float
    ) -> CoordinateSystem:
        """Creates and returns a CoordinateSystem instance from explicit values.

        This is a pure function that creates a CoordinateSystem object. All logic
        for determining the z-position based on previous surfaces has been moved
        to the state manager (SurfaceGroup).

        Args:
            x (float): X-coordinate.
            y (float): Y-coordinate.
            z (float): Z-coordinate.
            rx (float): Rotation about the X-axis in degrees.
            ry (float): Rotation about the Y-axis in degrees.
            rz (float): Rotation about the Z-axis in degrees.

        Returns:
            CoordinateSystem: The configured coordinate system instance.
        """
        return CoordinateSystem(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)
