from abc import ABC, abstractmethod
import warnings
import numpy as np
from optiland.geometries.standard import StandardGeometry


class NewtonRaphsonGeometry(StandardGeometry, ABC):
    """
    Represents a geometry that uses the Newton-Raphson method for ray tracing.

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

    def __init__(self, coordinate_system, radius, conic=0.0, tol=1e-10,
                 max_iter=100):
        super().__init__(coordinate_system, radius, conic)
        self.tol = tol
        self.max_iter = max_iter

    @abstractmethod
    def sag(self, x=0, y=0):
        """Calculate the surface sag of the geometry.

        Args:
            x (float or np.ndarray, optional): The x-coordinate. Defaults to 0.
            y (float or np.ndarray, optional): The y-coordinate. Defaults to 0.

        Returns:
            Union[float, np.ndarray]: The surface sag of the geometry.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _surface_normal(self, x, y):
        """Calculate the surface normal of the geometry at the given x and y
        position.

        Args:
            x (float, np.ndarray): The x values to use for calculation.
            y (float, np.ndarray): The y values to use for calculation.

        Returns:
            tuple: The surface normal components (nx, ny, nz).
        """
        pass  # pragma: no cover

    def surface_normal(self, rays):
        """
        Calculates the surface normal of the geometry at the given rays.

        Args:
            rays (Rays): The rays used for calculating the surface normal.

        Returns:
            tuple: The surface normal components (nx, ny, nz).
        """
        return self._surface_normal(rays.x, rays.y)

    def distance(self, rays):
        """
        Calculates the distance between the geometry and the given ray
        positions.

        Note:
            This function uses the Newton-Raphson method for ray tracing.

        Args:
            rays (Rays): The rays used for calculating distance.

        Returns:
            numpy.ndarray: The distances between the geometry and the rays.
        """
        x, y, z = self._intersection_sphere(rays)
        intersections = np.column_stack((x, y, z))
        ray_directions = np.column_stack((rays.L, rays.M, rays.N))
        for i in range(self.max_iter):
            z_surface = self.sag(intersections[:, 0], intersections[:, 1])
            dz = intersections[:, 2] - z_surface
            distance = dz / ray_directions[:, 2]
            intersections -= distance[:, None] * ray_directions
            if np.max(np.abs(dz)) < self.tol:
                break
        position = np.column_stack((rays.x, rays.y, rays.z))
        return np.linalg.norm(intersections - position, axis=1)

    def _intersection_sphere(self, rays):
        """
        Calculates the intersection points of the rays with the geometry.

        Args:
            rays (Rays): The rays to calculate the intersection points for.

        Returns:
            tuple: The intersection points (x, y, z).
        """
        a = rays.L**2 + rays.M**2 + rays.N**2
        b = (2 * rays.L * rays.x + 2 * rays.M * rays.y -
             2 * rays.N * self.radius + 2 * rays.N * rays.z)
        c = (rays.x**2 + rays.y**2 + rays.z**2 - 2 * self.radius * rays.z)

        # discriminant
        d = b ** 2 - 4 * a * c

        # two solutions for distance to sphere
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t1 = (-b + np.sqrt(d)) / (2 * a)
            t2 = (-b - np.sqrt(d)) / (2 * a)

        # intersections "behind" ray, set to inf to ignore
        t1[t1 < 0] = np.inf
        t2[t2 < 0] = np.inf

        # find intersection points in z
        z1 = rays.z + t1 * rays.N
        z2 = rays.z + t2 * rays.N

        # take intersection closest to z = 0 (i.e., vertex of geometry)
        t = np.where(np.abs(z1) <= np.abs(z2), t1, t2)

        # handle case when a = 0
        cond = a == 0
        t[cond] = -c[cond] / b[cond]

        x = rays.x + rays.L * t
        y = rays.y + rays.M * t
        z = rays.z + rays.N * t

        return x, y, z
