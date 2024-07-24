from abc import ABC, abstractmethod
import numpy as np
from optiland.rays import RealRays


class BaseBSDF(ABC):

    def scatter(self, rays: RealRays):
        """
        Scatter rays according to the BSDF.

        Args:
            rays (RealRays): The rays with direction cosines to be rotated.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The rotated direction
                cosines of the rays.
        """
        x, y = self._generate_points()

        # merge L, M, N into a single arrays
        v = np.column_stack((rays.L, rays.M, rays.N))  # ray directions

        # arbitrary vector to use as a reference for the cross product
        arbitrary_vector = np.array([1, 0, 0])
        aligned = np.isclose(v[:, 0], 1.0)
        arbitrary_vector = np.where(aligned[:, np.newaxis], [0, 1, 0],
                                    arbitrary_vector)

        # first basis vector for the new coordinate system
        a = np.cross(v, arbitrary_vector)
        a = a / np.linalg.norm(a, axis=1)[:, np.newaxis]

        # second basis vector for the new coordinate system
        b = np.cross(v, a)
        b = b / np.linalg.norm(b, axis=1)[:, np.newaxis]

        # generate scattered vectors
        v_scatter = v + x[:, np.newaxis] * a + y[:, np.newaxis] * b

        rays.L = v_scatter[:, 0]
        rays.M = v_scatter[:, 1]
        rays.N = v_scatter[:, 2]

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
