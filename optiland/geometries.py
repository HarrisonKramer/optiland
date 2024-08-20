"""Optiland Geometries Module

This module defines the base and specific geometrical shapes used in the
Optiland optical simulation package. It provides a framework for defining
various optical geometries, including their properties and behaviors such as
sag, ray distance calculation, and surface normals.

Kramer Harrison, 2024
"""
import abc
import warnings
import numpy as np


class BaseGeometry:
    """Base geometry for all geometries.

    Args:
        cs (CoordinateSystem): The coordinate system of the geometry.
    """

    def __init__(self, coordinate_system):
        self.cs = coordinate_system

    @abc.abstractmethod
    def sag(self, x=0, y=0):
        """Calculate the surface sag of the geometry.

        Args:
            x (float or np.ndarray, optional): The x-coordinate. Defaults to 0.
            y (float or np.ndarray, optional): The y-coordinate. Defaults to 0.

        Returns:
            Union[float, np.ndarray]: The surface sag of the geometry.
        """
        return

    @abc.abstractmethod
    def distance(self, rays):
        """Find the propagation distance to the geometry.

        Args:
            rays (RealRays): The rays to calculate the distance for.

        Returns:
            np.ndarray: The propagation distance to the geometry.
        """
        return

    @abc.abstractmethod
    def surface_normal(self, rays):
        """Find the surface normal of the geometry at the given ray positions.

        Args:
            rays (RealRays): The rays position at which to calculate the
                surface normal.

        Returns:
            np.ndarray: The surface normals of the geometry at the given
                ray positions.
        """
        return

    def localize(self, rays):
        """Convert rays from the global coordinate system to the local
        coordinate system.

        Args:
            rays (RealRays): The rays to convert.
        """
        self.cs.localize(rays)

    def globalize(self, rays):
        """Convert rays from the local coordinate system to the global
        coordinate system.

        Args:
            rays (RealRays): The rays to convert.
        """
        self.cs.globalize(rays)


class Plane(BaseGeometry):
    """An infinite plane geometry.

    Args:
        cs (CoordinateSystem): The coordinate system of the plane geometry.
    """

    def __init__(self, coordinate_system):
        super().__init__(coordinate_system)
        self.radius = np.inf

    def sag(self, x=0, y=0):
        """Calculate the surface sag of the plane geometry.

        Args:
            x (float or np.ndarray, optional): The x-coordinate of the point
                on the plane. Defaults to 0.
            y (float or np.ndarray, optional): The y-coordinate of the point
                on the plane. Defaults to 0.

        Returns:
            Union[float, np.ndarray]: The surface sag of the plane at the
                given point.
        """
        if isinstance(y, np.ndarray):
            return np.zeros_like(y)
        return 0

    def distance(self, rays):
        """Find the propagation distance to the plane geometry.

        Args:
            rays (RealRays): The rays used to calculate the distance.

        Returns:
            np.ndarray: The propagation distance to the plane geometry for
                each ray.
        """
        t = -rays.z / rays.N

        # if rays do not hit plane, set to NaN
        try:
            t[t < 0] = np.nan
        except TypeError:  # input is not an array
            if t < 0:
                t = np.nan

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


class StandardGeometry(BaseGeometry):
    """
    Represents a standard geometry with a given coordinate system, radius, and
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
        return r2 / (self.radius *
                     (1 + np.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))

    def distance(self, rays):
        """Find the propagation distance to the geometry for the given rays.

        Args:
            rays: The rays for which to calculate the distance.

        Returns:
            ndarray: The distances to the geometry.
        """
        a = self.k * rays.N**2 + rays.L**2 + rays.M**2 + rays.N**2
        b = (2 * self.k * rays.N * rays.z
             + 2 * rays.L * rays.x
             + 2 * rays.M * rays.y
             - 2 * rays.N * self.radius
             + 2 * rays.N * rays.z)
        c = (self.k * rays.z**2 - 2 * self.radius * rays.z
             + rays.x**2 + rays.y**2 + rays.z**2)

        # discriminant
        d = b ** 2 - 4 * a * c

        # two solutions for distance to conic
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

        denom = -self.radius * np.sqrt(1 - (1 + self.k)*r2 / self.radius**2)
        dfdx = rays.x / denom
        dfdy = rays.y / denom
        dfdz = 1

        mag = np.sqrt(dfdx**2 + dfdy**2 + dfdz**2)

        nx = dfdx / mag
        ny = dfdy / mag
        nz = dfdz / mag

        return nx, ny, nz


class EvenAsphere(StandardGeometry):
    # TODO: clean up format & improve docs on mathematics used
    """
    Represents an even asphere geometry.

    Args:
        coordinate_system (str): The coordinate system used for the geometry.
        radius (float): The radius of the asphere.
        conic (float, optional): The conic constant of the asphere.
            Defaults to 0.0.
        coefficients (list, optional): The coefficients of the asphere.
            Defaults to an empty list, indicating no aspheric coefficients are
            used.
        tol (float, optional): The tolerance value used in calculations.
            Defaults to 1e-10.
    """

    def __init__(self, coordinate_system, radius, conic=0.0, coefficients=[],
                 tol=1e-10):
        super().__init__(coordinate_system, radius, conic)
        self.c = coefficients
        self.tol = tol

    def sag(self, x=0, y=0):
        """
        Calculates the sag of the asphere at the given coordinates.

        Args:
            x (float, np.ndarray, optional): The x-coordinate(s).
                Defaults to 0.
            y (float, np.ndarray, optional): The y-coordinate(s).
                Defaults to 0.

        Returns:
            float: The sag value at the given coordinates.
        """
        r2 = x**2 + y**2
        z = r2 / (self.radius *
                  (1 + np.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))
        for i, Ci in enumerate(self.c):
            z += Ci * r2 ** (i + 1)

        return z

    def distance(self, rays):
        """
        Calculates the distance between the asphere and the given ray
        positions.

        Note:
            This function uses the Newton-Raphson method for tracing an
            asphere.

        Args:
            rays (Rays): The rays used for calculating distance.

        Returns:
            numpy.ndarray: The distances between the asphere and the rays.
        """
        xs, ys, zs = self._intersection_sphere(rays)

        x0 = xs - rays.L / rays.N * zs
        y0 = ys - rays.M / rays.N * zs

        zas0 = self.sag(xs, ys)

        x1 = xs
        y1 = ys
        z1 = zas0

        nx, ny, nz = self._surface_normal(x1, y1)

        z2 = np.ones_like(x0) * 1e10
        zas1 = np.zeros_like(x0)

        while np.max(np.abs(z2 - zas1)) > self.tol:
            dot = nx * rays.L + ny * rays.M + nz * rays.N
            a = (nx * (x1 - x0) + ny * (y1 - y0) + nz * z1) / dot

            z2 = a * rays.N
            x2 = x0 + rays.L / rays.N * z2
            y2 = y0 + rays.M / rays.N * z2

            zas1 = self.sag(x2, y2)

            x1 = x2
            y1 = y2
            z1 = zas1

            nx, ny, nz = self._surface_normal(x2, y2)

        return np.sqrt((rays.x - x2)**2 + (rays.y - y2)**2 + (rays.z - z2)**2)

    def surface_normal(self, rays):
        """
        Calculates the surface normal of the asphere at the given rays.

        Args:
            rays (Rays): The rays used for calculating the surface normal.

        Returns:
            tuple: The surface normal components (nx, ny, nz).
        """
        return self._surface_normal(rays.x, rays.y)

    def _surface_normal(self, x, y):
        """
        Calculates the surface normal of the asphere at the given x and y
        position.

        Args:
            x (np.ndarray): The x values to use for calculation.
            y (np.ndarray): The y values to use for calculation.

        Returns:
            tuple: The surface normal components (nx, ny, nz).
        """
        r2 = x**2 + y**2

        denom = np.sqrt(self.radius**2 - (1 + self.k) * r2)
        dfdx = x / denom
        dfdy = y / denom

        for i, Ci in enumerate(self.c):
            dfdx += 2 * (i+1) * x * Ci * r2**i
            dfdy += 2 * (i+1) * y * Ci * r2**i

        mag = np.sqrt(dfdx**2 + dfdy**2 + 1)

        nx = -dfdx / mag
        ny = -dfdy / mag
        nz = 1 / mag

        return nx, ny, nz

    def _intersection_sphere(self, rays):
        """
        Calculates the intersection points of the rays with the asphere.

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
        t[a == 0] = -c[a == 0] / b[a == 0]

        x = rays.x + rays.L * t
        y = rays.y + rays.M * t
        z = rays.z + rays.N * t

        return x, y, z


class PolynomialGeometry(BaseGeometry):
    """
    Represents a polynomial geometry defined as:

    z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Cij * x^i * y^j)

    where
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - Cij are the polynomial coefficients

    The coefficients are defined in a 2D array where coefficients[i][j] is the
    coefficient for x^i * y^j.

    Args:
        coordinate_system (str): The coordinate system used for the geometry.
        radius (float): The radius of the asphere.
        conic (float, optional): The conic constant of the asphere.
            Defaults to 0.0.
        coefficients (list, optional): The coefficients of the asphere.
            Defaults to an empty list, indicating no aspheric coefficients are
            used.
        tol (float, optional): The tolerance value used in calculations.
            Defaults to 1e-10.
        max_iter (int, optional): The maximum number of iterations used in
            calculations. Defaults to 100.
    """

    def __init__(self, coordinate_system, radius, conic=0.0, coefficients=[],
                 tol=1e-10, max_iter=100):
        super().__init__(coordinate_system)
        self.coefficients = coefficients
        self.radius = radius
        self.k = conic
        self.tol = tol
        self.max_iter = max_iter

    def sag(self, x=0, y=0):
        """
        Calculates the sag of the polynomial surface at the given coordinates.

        Args:
            x (float, np.ndarray, optional): The x-coordinate(s).
                Defaults to 0.
            y (float, np.ndarray, optional): The y-coordinate(s).
                Defaults to 0.

        Returns:
            float: The sag value at the given coordinates.
        """
        r2 = x**2 + y**2
        z = r2 / (self.radius *
                  (1 + np.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))
        for i in range(len(self.coefficients)):
            for j in range(len(self.coefficients[i])):
                z += self.coefficients[i][j] * (x ** i) * (y ** j)
        return z

    def distance(self, rays):
        """
        Calculates the distance between the polynomial geometry and the given
        ray positions.

        Note:
            This function uses the Newton-Raphson method.

        Args:
            rays (Rays): The rays used for calculating distance.

        Returns:
            numpy.ndarray: The distances between the asphere and the rays.
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

    def surface_normal(self, rays):
        """
        Calculates the surface normal of the polynomial surface for the given
        rays.

        Args:
            rays (Rays): The rays used for calculating the surface normal.

        Returns:
            tuple: The surface normal components (nx, ny, nz).
        """
        n = self._surface_normal(rays.x, rays.y)
        return n

    def _surface_normal(self, x, y):
        """
        Calculates the surface normal of the polynomial surface at the given x
        and y position.

        Args:
            x (np.ndarray): The x values to use for calculation.
            y (np.ndarray): The y values to use for calculation.

        Returns:
            tuple: The surface normal components (nx, ny, nz).
        """
        r2 = x**2 + y**2
        denom = -self.radius * np.sqrt(1 - (1 + self.k)*r2 / self.radius**2)
        dzdx = x / denom
        dzdy = y / denom

        for i in range(1, len(self.coefficients)):
            for j in range(len(self.coefficients[i])):
                dzdx -= i * self.coefficients[i][j] * (x ** (i - 1)) * (y ** j)

        for i in range(len(self.coefficients)):
            for j in range(1, len(self.coefficients[i])):
                dzdy -= j * self.coefficients[i][j] * (x ** i) * (y ** (j - 1))

        norm = np.sqrt(dzdx**2 + dzdy**2 + 1)
        nx = dzdx / norm
        ny = dzdy / norm
        nz = 1 / norm

        return nx, ny, nz

    def _intersection_sphere(self, rays):
        """
        Calculates the intersection points of the rays with a sphere.

        Args:
            rays (Rays): The rays used for calculating the intersection points.

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
        t[a == 0] = -c[a == 0] / b[a == 0]

        x = rays.x + rays.L * t
        y = rays.y + rays.M * t
        z = rays.z + rays.N * t

        return x, y, z
