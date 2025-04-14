"""Newton Raphson Geometry

The Newton Raphson geometry represents a surface utilizing the Newton-Raphson
method for ray tracing. This is an abstract base class that should be inherited
by any geometry that uses the Newton-Raphson method for ray tracing.

Kramer Harrison, 2024
"""

import warnings
from abc import ABC, abstractmethod

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.standard import StandardGeometry


class NewtonRaphsonGeometry(StandardGeometry, ABC):
    """Represents a geometry that uses the Newton-Raphson method for ray tracing.

    Args:
        coordinate_system (str): The coordinate system used for the geometry.
        radius (float): The radius of curvature of the geometry.
        conic (float, optional): The conic constant of the geometry.
            Defaults to 0.0.
        tol (float, optional): The tolerance value used in calculations.
            Defaults to 1e-10.
        max_iter (int, optional): The maximum number of iterations used in
            calculations. Defaults to 100.

    """

    def __init__(self, coordinate_system, radius, conic=0.0, tol=1e-10, max_iter=100):
        super().__init__(coordinate_system, radius, conic)
        self.tol = tol
        self.max_iter = max_iter

    def __str__(self):
        return "Newton Raphson"  # pragma: no cover

    @abstractmethod
    def sag(self, x=0, y=0):
        """Calculate the surface sag of the geometry.

        Args:
            x (float or be.ndarray, optional): The x-coordinate. Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate. Defaults to 0.

        Returns:
            Union[float, be.ndarray]: The surface sag of the geometry.

        """
        # pragma: no cover

    @abstractmethod
    def _surface_normal(self, x, y):
        """Calculate the surface normal of the geometry at the given x and y
        position.

        Args:
            x (float, be.ndarray): The x values to use for calculation.
            y (float, be.ndarray): The y values to use for calculation.

        Returns:
            tuple: The surface normal components (nx, ny, nz).

        """
        # pragma: no cover

    def surface_normal(self, rays):
        """Calculates the surface normal of the geometry at the given rays.

        Args:
            rays (Rays): The rays used for calculating the surface normal.

        Returns:
            tuple: The surface normal components (nx, ny, nz).

        """
        return self._surface_normal(rays.x, rays.y)

    def distance(self, rays):
        """Calculates the distance between the geometry and the given ray
        positions.

        Note:
            This function uses the Newton-Raphson method for ray tracing.

        Args:
            rays (Rays): The rays used for calculating distance.

        Returns:
            numpy.ndarray: The distances between the geometry and the rays.

        """
        x, y, z = self._intersection_sphere(rays)
        intersections = be.column_stack((x, y, z))
        ray_directions = be.column_stack((rays.L, rays.M, rays.N))

        for _ in range(self.max_iter):
            z_surface = self.sag(intersections[:, 0], intersections[:, 1])
            dz = intersections[:, 2] - z_surface
            distance = dz / ray_directions[:, 2]
            intersections -= distance[:, None] * ray_directions
            if be.max(be.abs(dz)) < self.tol:
                break

        position = be.column_stack((rays.x, rays.y, rays.z))
        return be.linalg.norm(intersections - position, axis=1)

    def _intersection_sphere(self, rays):
        """Calculates the intersection points of the rays with the geometry.

        Args:
            rays (Rays): The rays to calculate the intersection points for.

        Returns:
            tuple: The intersection points (x, y, z).

        """
        a = rays.L**2 + rays.M**2 + rays.N**2
        b = (
            2 * rays.L * rays.x
            + 2 * rays.M * rays.y
            - 2 * rays.N * self.radius
            + 2 * rays.N * rays.z
        )
        c = rays.x**2 + rays.y**2 + rays.z**2 - 2 * self.radius * rays.z

        # discriminant
        d = b**2 - 4 * a * c

        # two solutions for distance to sphere
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t1 = (-b + be.sqrt(d)) / (2 * a)
            t2 = (-b - be.sqrt(d)) / (2 * a)

        # find intersection points in z
        z1 = rays.z + t1 * rays.N
        z2 = rays.z + t2 * rays.N

        # take intersection closest to z = 0 (i.e., vertex of geometry)
        t = be.where(be.abs(z1) <= be.abs(z2), t1, t2)

        # handle case when a = 0
        cond = a == 0
        t[cond] = -c[cond] / b[cond]

        x = rays.x + rays.L * t
        y = rays.y + rays.M * t
        z = rays.z + rays.N * t

        return x, y, z

    def to_dict(self):
        """Converts the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        geometry_dict = super().to_dict()
        geometry_dict.update({"tol": self.tol, "max_iter": self.max_iter})
        return geometry_dict

    @classmethod
    def from_dict(cls, data):  # pragma: no cover
        """Creates a geometry from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the geometry.

        Returns:
            NewtonRaphsonGeometry: The geometry created from the dictionary.

        """
        required_keys = {"cs", "radius"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])
        conic = data.get("conic", 0.0)
        tol = data.get("tol", 1e-10)
        max_iter = data.get("max_iter", 100)

        return cls(cs, data["radius"], conic, tol, max_iter)
