"""Standard Geometry

The Standard geometry represents a surface defined by a sphere or conic in two
dimensions. The surface is defined as:

z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2)))

where
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant

Kramer Harrison, 2024
"""

import warnings

import numpy as np

from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.base import BaseGeometry


class StandardGeometry(BaseGeometry):
    """Represents a standard geometry with a given coordinate system, radius, and
    conic.

    Args:
        coordinate_system (str): The coordinate system of the geometry.
        radius (float): The radius of the geometry.
        conic (float): The conic value of the geometry.

    Methods:
        sag(x=0, y=0): Calculates the surface sag of the geometry at the given
            coordinates.
        distance(rays): Finds the propagation distance to the geometry for the
            given rays.
        surface_normal(rays): Calculates the surface normal of the geometry at
            the given ray positions.

    """

    def __init__(self, coordinate_system, radius, conic=0.0):
        super().__init__(coordinate_system)
        self.radius = radius
        self.k = conic
        self.is_symmetric = True

    def __str__(self):
        return "Standard"

    def sag(self, x=0, y=0):
        """Calculate the surface sag of the geometry at the given coordinates.

        Args:
            x (float, np.ndarray, optional): The x-coordinate(s).
                Defaults to 0.
            y (float, np.ndarray, optional): The y-coordinate(s).
                Defaults to 0.

        Returns:
            float: The sag value at the given coordinates.

        """
        r2 = x**2 + y**2
        return r2 / (
            self.radius * (1 + np.sqrt(1 - (1 + self.k) * r2 / self.radius**2))
        )

    def distance(self, rays):
        """Find the propagation distance to the geometry for the given rays.

        Args:
            rays: The rays for which to calculate the distance.

        Returns:
            ndarray: The distances to the geometry.

        """
        a = self.k * rays.N**2 + rays.L**2 + rays.M**2 + rays.N**2
        b = (
            2 * self.k * rays.N * rays.z
            + 2 * rays.L * rays.x
            + 2 * rays.M * rays.y
            - 2 * rays.N * self.radius
            + 2 * rays.N * rays.z
        )
        c = (
            self.k * rays.z**2
            - 2 * self.radius * rays.z
            + rays.x**2
            + rays.y**2
            + rays.z**2
        )

        # discriminant
        d = b**2 - 4 * a * c

        # two solutions for distance to conic
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t1 = (-b + np.sqrt(d)) / (2 * a)
            t2 = (-b - np.sqrt(d)) / (2 * a)

        # find intersection points in z
        z1 = rays.z + t1 * rays.N
        z2 = rays.z + t2 * rays.N

        # take intersection closest to z = 0 (i.e., vertex of geometry)
        t = np.where(np.abs(z1) <= np.abs(z2), t1, t2)

        # handle case when a = 0
        t[a == 0] = -c[a == 0] / b[a == 0]

        return t

    def surface_normal(self, rays):
        """Calculate the surface normal of the geometry at the given points.

        Args:
            rays: The ray positions at which to calculate the surface normals.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The x, y, and z
                components of the surface normal.

        """
        r2 = rays.x**2 + rays.y**2

        denom = self.radius * np.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        dfdx = rays.x / denom
        dfdy = rays.y / denom
        dfdz = -1

        mag = np.sqrt(dfdx**2 + dfdy**2 + dfdz**2)

        nx = dfdx / mag
        ny = dfdy / mag
        nz = dfdz / mag

        return nx, ny, nz

    def to_dict(self):
        """Convert the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        geometry_dict = super().to_dict()
        geometry_dict.update({"radius": self.radius, "conic": self.k})
        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        """Create a geometry from a dictionary.

        Args:
            data (dict): The dictionary representation of the geometry.

        Returns:
            StandardGeometry: The geometry.

        """
        required_keys = {"cs", "radius"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])

        return cls(cs, data["radius"], data.get("conic", 0.0))
