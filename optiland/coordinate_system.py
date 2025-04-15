"""Coordinate System Module

This module provides standard coordinate system transformation calculations.
The CoordinateSystem class represents a coordinate system in 3D space and
provides methods for localizing and globalizing rays. This class is used
to define the position and orientation of all optical surfaces in an optical
system.

Kramer Harrison, 2024
"""

from scipy.spatial.transform import Rotation as R

import optiland.backend as be
from optiland.rays import RealRays


class CoordinateSystem:
    """Represents a coordinate system in 3D space.

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

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        rx: float = 0,
        ry: float = 0,
        rz: float = 0,
        reference_cs: "CoordinateSystem" = None,
    ):
        self.x = x
        self.y = y
        self.z = z

        self.rx = rx
        self.ry = ry
        self.rz = rz

        self.reference_cs = reference_cs

    def localize(self, rays):
        """Localizes the rays in the coordinate system.

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
        """Globalizes the rays from the coordinate system.

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
        """Returns the position of the coordinate system in the global coordinate
            system.

        Returns:
            tuple: The x, y, and z coordinates of the position.

        """
        vector = RealRays(0, 0, 0, 0, 0, 1, 1, 1)
        self.globalize(vector)
        return vector.x, vector.y, vector.z

    def get_rotation_matrix(self):
        """Get the rotation matrix of the coordinate system

        Returns:
            be.ndarray: The rotation matrix of the coordinate system.

        """
        rotation = be.array([self.rx, self.ry, self.rz])
        return R.from_euler("xyz", rotation).as_matrix()

    def get_effective_transform(self):
        """Get the effective translation and rotation matrix of the CS

        Returns:
            tuple: The effective translation and rotation matrix

        """
        translation = be.array([self.x, self.y, self.z])
        if self.reference_cs is None:
            # No reference coordinate system, return the local transform
            return translation, self.get_rotation_matrix()

        # Get the effective transform of the reference coordinate system
        ref_translation, ref_rot_mat = self.reference_cs.get_effective_transform()

        # Combine translations
        eff_translation = ref_translation + ref_rot_mat @ translation

        # Combine rotations by multiplying the rotation matrices
        eff_rot_mat = ref_rot_mat @ self.get_rotation_matrix()

        return eff_translation, eff_rot_mat

    def get_effective_rotation_euler(self):
        """Get the effective rotation in Euler angles

        Returns:
            be.ndarray: The effective rotation in Euler angles

        """
        _, eff_rot_mat = self.get_effective_transform()
        # Convert the effective rotation matrix back to Euler angles
        return R.from_matrix(eff_rot_mat).as_euler("xyz")

    def to_dict(self):
        """Convert the coordinate system to a dictionary.

        Returns:
            dict: The dictionary representation of the coordinate system.

        """
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "rx": float(self.rx),
            "ry": float(self.ry),
            "rz": float(self.rz),
            "reference_cs": self.reference_cs.to_dict() if self.reference_cs else None,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a coordinate system from a dictionary.

        Args:
            data (dict): The dictionary representation of the coordinate
                system.

        Returns:
            CoordinateSystem: The coordinate system.

        """
        reference_cs = (
            cls.from_dict(data["reference_cs"]) if data["reference_cs"] else None
        )

        return cls(
            data.get("x", 0),
            data.get("y", 0),
            data.get("z", 0),
            data.get("rx", 0),
            data.get("ry", 0),
            data.get("rz", 0),
            reference_cs,
        )
