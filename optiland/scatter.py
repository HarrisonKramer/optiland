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
        L, M = self._generate_points()
        N = np.sqrt(1 - L**2 - M**2)

        # merge L, M, N into a single arrays
        s = np.column_stack((rays.L, rays.M, rays.N))  # scatter directions
        v = np.column_stack((L, M, N))  # ray directions

        # z axis is reference for rotation
        z = np.array([0.0, 0.0, 1.0])

        # find the axis of rotation
        k = np.cross(z, v)
        k_norm = np.linalg.norm(k, axis=1, keepdims=True)
        k = np.divide(k, k_norm, where=k_norm != 0)

        # precompute sin and cos of the angle of rotation
        cos_theta = np.dot(z, v)
        sin_theta = np.sqrt(1 - cos_theta**2)

        # Rodrigues' rotation formula
        dot_ks = np.einsum('ij,ij->i', k, s)
        cross_ks = np.cross(k, s)

        L_new = (s[:, 0] * cos_theta + cross_ks[:, 0] * sin_theta +
                 k[:, 0] * dot_ks * (1 - cos_theta))
        M_new = (s[:, 1] * cos_theta + cross_ks[:, 1] * sin_theta +
                 k[:, 1] * dot_ks * (1 - cos_theta))
        N_new = (s[:, 2] * cos_theta + cross_ks[:, 2] * sin_theta +
                 k[:, 2] * dot_ks * (1 - cos_theta))

        # handle case when vector already aligned with z axis
        cond = (k_norm == 0).squeeze()
        L_new[cond] = v[cond, 0]
        M_new[cond] = v[cond, 1]
        N_new[cond] = v[cond, 2]

        return L_new, M_new, N_new

    @abstractmethod
    def _generate_points(self):
        """Generate points on the unit disk."""
        pass


def LambertianBSDF(BaseBSDF):

    def _generate_points(self, rays: RealRays):
        r = np.random.rand(rays.x.size)
        theta = np.random.uniform(0, 2 * np.pi, rays.x.size)
        L = np.sqrt(r) * np.cos(theta)
        M = np.sqrt(r) * np.sin(theta)
        return L, M
