"""Optiland Coordinate System Module

This module provides standard coordinate system transformation calculations

Kramer Harrison, 2024
"""
from optiland.rays import RealRays


class CoordinateSystem:
    """
    Represents a coordinate system in 3D space.

    Args:
        x (float): The x-coordinate of the origin.
        y (float): The y-coordinate of the origin.
        z (float): The z-coordinate of the origin.
        rx (float): The rotation around the x-axis.
        ry (float): The rotation around the y-axis.
        rz (float): The rotation around the z-axis.
        reference_cs (CoordinateSystem): The reference coordinate system.

    Attributes:
        x (float): The x-coordinate of the origin.
        y (float): The y-coordinate of the origin.
        z (float): The z-coordinate of the origin.
        rx (float): The rotation around the x-axis.
        ry (float): The rotation around the y-axis.
        rz (float): The rotation around the z-axis.
        reference_cs (CoordinateSystem): The reference coordinate system.

    """

    def __init__(self, x: float = 0, y: float = 0, z: float = 0,
                 rx: float = 0, ry: float = 0, rz: float = 0,
                 reference_cs: 'CoordinateSystem' = None):
        self.x = x
        self.y = y
        self.z = z

        self.rx = rx
        self.ry = ry
        self.rz = rz

        self.reference_cs = reference_cs

    def localize(self, rays):
        """
        Localizes the rays in the coordinate system.

        Args:
            rays: The rays to be localized.

        """
        if self.reference_cs:
            self.reference_cs.localize(rays)

        rays.translate(-self.x, -self.y, -self.z)
        if self.rx:
            rays.rotate_x(-self.rx)
        if self.ry:
            rays.rotate_y(-self.ry)
        if self.rz:
            rays.rotate_z(-self.rz)

    def globalize(self, rays):
        """
        Globalizes the rays from the coordinate system.

        Args:
            rays: The rays to be globalized.

        """
        if self.rz:
            rays.rotate_z(self.rz)
        if self.ry:
            rays.rotate_y(self.ry)
        if self.rx:
            rays.rotate_x(self.rx)
        rays.translate(self.x, self.y, self.z)

        if self.reference_cs:
            self.reference_cs.globalize(rays)

    @property
    def position_in_gcs(self):
        """
        Returns the position of the coordinate system in the global coordinate
            system.

        Returns:
            tuple: The x, y, and z coordinates of the position.

        """
        vector = RealRays(0, 0, 0, 0, 0, 1, 1, 1)
        self.globalize(vector)
        return vector.x, vector.y, vector.z
