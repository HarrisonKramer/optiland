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
        super().__init__(coordinate_system, radius)
        self.k = conic

    def sag(self, x=0, y=0):
        """Calculate the surface sag of the geometry at the given coordinates.

        Args:
            x (float or np.ndarray, optional): The x-coordinate. Defaults to 0.
            y (float or np.ndarray, optional): The y-coordinate. Defaults to 0.

        Returns:
            Union[float, np.ndarray]: The surface sag of the geometry.
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

        # take intersection closest to z = 0
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
