import abc
import warnings
import numpy as np


class BaseGeometry:
    """Base geometry for all geometries"""

    def __init__(self, coordinate_system, radius=np.inf):
        self.cs = coordinate_system
        self.radius = radius

    @abc.abstractmethod
    def sag(self, x=0, y=0):
        """Surface sag of geometry"""
        return

    @abc.abstractmethod
    def distance(self, rays):
        """Find propagation distance to geometry"""
        return

    @abc.abstractmethod
    def surface_normal(self, points):
        """Find surface normal of geometry at points"""
        return

    def localize(self, rays):
        """Convert rays from global coordinate system to local coordinate
        system"""
        self.cs.localize(rays)

    def globalize(self, rays):
        """Convert rays from local coordinate system to global coordinate
        system"""
        self.cs.globalize(rays)


class Plane(BaseGeometry):
    """An infinite plane geometry"""

    def __init__(self, coordinate_system):
        BaseGeometry.__init__(self, coordinate_system)

    def sag(self, x=0, y=0):
        """Surface sag of geometry"""
        if isinstance(y, np.ndarray):
            return np.zeros_like(y)
        return 0

    def distance(self, rays):
        """Find propagation distance to geometry"""
        t = -rays.z / rays.N

        # if rays do not hit plane, set to NaN
        try:
            t[t < 0] = np.nan
        except TypeError:  # input is not an array
            if t < 0:
                t = np.nan

        return t

    def surface_normal(self, rays):
        """Find surface normal of geometry at points"""
        return 0, 0, 1


class StandardGeometry(BaseGeometry):

    def __init__(self, coordinate_system, radius, conic=0.0):
        super().__init__(coordinate_system, radius)
        self.k = conic

    def sag(self, x=0, y=0):
        """Surface sag of geometry"""
        r2 = x**2 + y**2
        return r2 / (self.radius * (1 + np.sqrt(1 - (1 + self.k) * r2 /
                                                self.radius**2)))

    def distance(self, rays):
        """Find propagation distance to geometry"""
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
        """Find surface normal of geometry at points"""
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
