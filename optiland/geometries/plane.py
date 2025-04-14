"""Plane Geometry

The Plane geometry represents an infinite plane in two dimensions. The surface
is defined as an XY plane with z=0 for all points. Recall that surfaces are
always defined in the local coordinate system of the geometry.

Kramer Harrison, 2024
"""

import warnings

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.base import BaseGeometry


class Plane(BaseGeometry):
    """An infinite plane geometry.

    Args:
        cs (CoordinateSystem): The coordinate system of the plane geometry.

    """

    def __init__(self, coordinate_system):
        super().__init__(coordinate_system)
        self.radius = be.inf
        self.is_symmetric = True

    def __str__(self):
        return "Planar"

    def sag(self, x=0, y=0):
        """Calculate the surface sag of the plane geometry.

        Args:
            x (float or be.ndarray, optional): The x-coordinate of the point
                on the plane. Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate of the point
                on the plane. Defaults to 0.

        Returns:
            Union[float, be.ndarray]: The surface sag of the plane at the
                given point.

        """
        if isinstance(y, be.ndarray):
            return be.zeros_like(y)
        return 0

    def distance(self, rays):
        """Find the propagation distance to the plane geometry.

        Args:
            rays (RealRays): The rays used to calculate the distance.

        Returns:
            be.ndarray: The propagation distance to the plane geometry for
                each ray.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = -rays.z / rays.N

        return t

    def surface_normal(self, rays):
        """Find the surface normal of the plane geometry at the given points.

        Args:
            rays (RealRays): The rays used to calculate the surface normal.

        Returns:
            Tuple[float, float, float]: The surface normal of the plane
                geometry at each point.

        """
        return 0, 0, 1

    def to_dict(self):
        """Convert the plane geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the plane geometry.

        """
        geometry_dict = super().to_dict()
        geometry_dict.update(
            {
                "radius": be.inf,
            },
        )
        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        """Create a plane geometry from a dictionary.

        Args:
            data (dict): The dictionary representation of the plane geometry.

        Returns:
            Plane: The plane geometry.

        """
        cs = CoordinateSystem.from_dict(data["cs"])
        return cls(cs)
