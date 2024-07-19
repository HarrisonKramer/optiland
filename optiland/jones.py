from abc import ABC, abstractmethod
import numpy as np
from optiland.rays import RealRays


class BaseJones(ABC):
    """Base class for Jones matrices.

    The class defines Jones matrices given ray properties. In the general case,
    the Jones matrix is an Nx3x3 array, where N is the number of rays. The
    array is padded to make it 3x3 to account for 3D ray calculations.
    """

    @abstractmethod
    def calculate_matrix(self, rays: RealRays, reflect: bool = False,
                         nx: np.ndarray = None, ny: np.ndarray = None,
                         nz: np.ndarray = None, aoi: np.ndarray = None):
        """
        Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            nx (np.ndarray, optional): Array representing the x-component of
                the surface normal vector. Defaults to None.
            ny (np.ndarray, optional): Array representing the y-component of
                the surface normal vector. Defaults to None.
            nz (np.ndarray, optional): Array representing the z-component of
                the surface normal vector. Defaults to None.
            aoi (np.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            np.ndarray: The calculated Jones matrix.
        """
        return np.tile(np.eye(3), (rays.x.size, 1, 1))


class JonesFresnel(BaseJones):
    """
    Class representing the Jones matrix for Fresnel calculations.

    Args:
        material_pre (Material): Material object representing the
            material before the surface.
        material_post (Material): Material object representing the
            material after the surface.
    """

    def __init__(self, material_pre, material_post):
        self.material_pre = material_pre
        self.material_post = material_post

    def calculate_matrix(self, rays: RealRays, reflect: bool = False,
                         aoi: np.ndarray = None):
        """
        Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            aoi (np.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            np.ndarray: The calculated Jones matrix.
        """
        # define local variables
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)

        # precomputations for speed
        cos_theta_i = np.cos(aoi)
        n = n2 / n1
        radicand = (n**2 - np.sin(aoi)**2).astype(complex)
        root = np.sqrt(radicand)

        # compute fresnel coefficients & compute jones matrices
        jones_matrix = np.zeros((rays.x.size, 3, 3), dtype=complex)
        if reflect:
            s = (cos_theta_i - root) / (cos_theta_i + root)
            p = (n**2*cos_theta_i - root) / (n**2*cos_theta_i + root)

            jones_matrix[:, 0, 0] = s
            jones_matrix[:, 1, 1] = -p
            jones_matrix[:, 2, 2] = -1
        else:
            s = 2 * cos_theta_i / (cos_theta_i + root)
            p = 2 * n * cos_theta_i / (n**2 * cos_theta_i + root)

            jones_matrix[:, 0, 0] = s
            jones_matrix[:, 1, 1] = p
            jones_matrix[:, 2, 2] = 1

        return jones_matrix
