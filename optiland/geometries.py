import abc
import numpy as np


class BaseGeometry:
    """Base geometry for all geometries"""

    def __init__(self, coordinate_system, radius=np.inf):
        self.cs = coordinate_system
        self.radius = radius

    def localize(self, rays):
        """Convert rays from global coordinate system to local coordinate
        system"""
        self.cs.localize(rays)

    def globalize(self, rays):
        """Convert rays from local coordinate system to global coordinate
        system"""
        self.cs.globalize(rays)

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

        # Discriminant
        d = b ** 2 - 4 * a * c

        # Use quadratic formula to solve for t
        t = (-b - np.sqrt(d)) / (2 * a)
        try:
            t[t < 0] = (-b[t < 0] + np.sqrt(d[t < 0])) / (2 * a[t < 0])
        except TypeError:  # input is not an array
            if t < 0:
                t = (-b + np.sqrt(d)) / (2 * a)

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
