import abc
import numpy as np


class BaseGeometry:

    def __init__(self, coordinate_system, radius=np.inf):
        self.cs = coordinate_system
        self.radius = radius

    def localize(self, rays):
        """Transform rays from global coordinate system to local coordinate system

        :param rays: rays to be transformed
        """
        self.cs.localize(rays)

    def globalize(self, rays):
        """Transform rays back from local coordinate system to global coordinate system

        :param rays: rays to be transformed
        """
        self.cs.globalize(rays)

    @abc.abstractmethod
    def sag(self, x=0, y=0):
        return

    @abc.abstractmethod
    def distance(self, rays):
        """Find parametric variable t for given rays and object.

        :param rays: rays to be transformed
        """
        return

    @abc.abstractmethod
    def surface_normal(self, points):
        """Find surface normal of object at each ray point (x, y, z)

        :param points: points on the surface at which surface normal will be computed
        """
        return


class Plane(BaseGeometry):
    """A plane geometry

    The plane has a normal vector (0, 0, 1) and origin (0, 0, 0) in the local coordinate, but it can be shifted
    and/or rotated by the provided coordinate system input
    """

    def __init__(self, coordinate_system):
        BaseGeometry.__init__(self, coordinate_system)

    def sag(self, x=0, y=0):
        if isinstance(y, np.ndarray):
            return np.zeros_like(y)
        return 0

    def distance(self, rays):
        """Find the distance t from the ray to the plane

        Note that find_t is always called in the local coordinate system of the plane, i.e. the plane normal will
        always be (0, 0, 1) and the point (0, 0, 0) will always lie on the plane (it is the origin). This implies the
        equation of the plane is f(x,y,z) = z = 0. In this case, the parametric variable will be t = -rays.z/rays.N

        :param rays: rays for which parametric distance t should be found
        """
        t = -rays.z / rays.N

        # Remove rays facing away from plane
        try:
            t[t < 0] = np.nan
        except TypeError:  # input is not an array
            if t < 0:
                t = np.nan

        return t

    def surface_normal(self, rays):
        """Find surface normal of the plane at each ray point (x, y, z). This is (0, 0, 1) in all cases.

        :param points: points on the surface at which surface normal will be computed
        """
        return 0, 0, 1


class StandardGeometry(BaseGeometry):

    def __init__(self, coordinate_system, radius, conic=0.0):
        super().__init__(coordinate_system, radius)
        self.k = conic

    def sag(self, x=0, y=0):
        r2 = x**2 + y**2
        return r2 / (self.radius * (1 + np.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))

    def distance(self, rays):
        """To find propagation distance, we plug in vector equations into surface sag equation"""
        a = self.k * rays.N**2 + rays.L**2 + rays.M**2 + rays.N**2
        b = 2 * self.k * rays.N * rays.z + 2 * rays.L * rays.x + 2 * rays.M * rays.y - \
            2 * rays.N * self.radius + 2 * rays.N * rays.z
        c = self.k * rays.z**2 - 2 * self.radius * rays.z + rays.x**2 + rays.y**2 + rays.z**2

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
