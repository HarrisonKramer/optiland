from abc import ABC, abstractmethod
import numpy as np
from optiland.rays import RealRays


class BaseBSDF(ABC):

    def scatter(self, rays: RealRays, nx: np.ndarray = None,
                ny: np.ndarray = None, nz: np.ndarray = None):
        """
        Scatter rays according to the BSDF.

        Args:
            rays (RealRays): The rays to be scattered.
            nx (np.ndarray): The x-component of the surface normal vector.
            ny (np.ndarray): The y-component of the surface normal vector.
            nz (np.ndarray): The z-component of the surface normal vector.

        Returns:
            RealRays: The updated rays after scattering is applied.
        """
        # generate scattered vectors
        x, y = self._generate_points()
        p = np.column_stack((x, y, np.zeros_like(x)))

        # merge surface normal vectors
        n = np.column_stack((nx, ny, nz))

        # merge ray vectors
        r = np.column_stack((rays.L, rays.M, rays.N))

        # arbitrary vector to use as a reference for the cross product
        arbitrary_vector = np.array([1, 0, 0])
        aligned = np.isclose(n[:, 0], 1.0)
        arbitrary_vector = np.where(aligned[:, np.newaxis], [0, 1, 0],
                                    arbitrary_vector)

        # first basis vector for the local coordinate system
        a = np.cross(n, arbitrary_vector)
        a = a / np.linalg.norm(a, axis=1)[:, np.newaxis]

        # second basis vector for the local coordinate system
        b = np.cross(n, a)

        # ray vector in local coordinate system
        r_loc = np.column_stack((np.dot(r, a), np.dot(r, b), np.dot(r, n)))

        # generate scattered vectors in local coordinate system
        v_scatter_loc = r_loc + p
        v_scatter_loc = (v_scatter_loc /
                         np.linalg.norm(v_scatter_loc, axis=1)[:, np.newaxis])

        # scatted vectors in global coordinate system
        v_scatter = np.dot(v_scatter_loc, np.column_stack((a, b, n)))

        # assign to rays
        rays.L = v_scatter[:, 0]
        rays.M = v_scatter[:, 1]
        rays.N = v_scatter[:, 2]

        return rays

    @abstractmethod
    def _generate_points(self, rays: RealRays):
        """Generate points on the unit disk.

        Args:
            rays (RealRays): The rays to be scattered.
        """
        pass


def LambertianBSDF(BaseBSDF):
    """
    Lambertian Bidirectional Scattering Distribution Function (BSDF) class.

    This class represents a Lambertian BSDF, which is generally used to model
    diffuse scattering.
    """

    def _generate_points(self, rays: RealRays):
        """Generate points on the unit disk.

        Args:
            rays (RealRays): The rays to be scattered.
        """
        r = np.random.rand(rays.x.size)
        theta = np.random.uniform(0, 2 * np.pi, rays.x.size)
        L = np.sqrt(r) * np.cos(theta)
        M = np.sqrt(r) * np.sin(theta)
        return L, M
