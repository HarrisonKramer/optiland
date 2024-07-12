from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from optiland.rays import RealRays


@dataclass
class InteractionParams:
    """
    Represents the parameters for an interaction between rays and a coating.

    Attributes:
        rays (RealRays): The rays involved in the interaction.
        aoi (np.ndarray): The angle of incidence of the rays (optional).
        L0 (np.ndarray): The L0 vector of the rays (optional).
        M0 (np.ndarray): The M0 vector of the rays (optional).
        N0 (np.ndarray): The N0 vector of the rays (optional).
    """
    rays: RealRays = None
    aoi: Optional[np.ndarray] = None
    L0: Optional[np.ndarray] = None
    M0: Optional[np.ndarray] = None
    N0: Optional[np.ndarray] = None


class BaseCoating(ABC):
    """
    Base class for coatings.

    This class defines the basic structure and behavior of a coating.

    Methods:
        interact: Performs an interaction based on the given parameters.
        reflect: Abstract method to handle reflection interaction.
        transmit: Abstract method to handle transmission interaction.
    """

    def interact(self, params: InteractionParams, reflect: bool = False):
        """
        Performs an interaction based on the given parameters.

        Args:
            params (InteractionParams): The parameters for the interaction.
            reflect (bool, optional): Flag indicating whether to perform
                reflection. Defaults to False.

        Returns:
            rays (RealRays): The rays after the interaction.
        """
        if reflect:
            return self.reflect(params)
        else:
            return self.transmit(params)

    def create_interaction_params(self, rays, nx=None, ny=None, nz=None):
        """
        Create interaction parameters for the given rays.

        Args:
            rays (RealRays): A list of rays for which interaction parameters
                need to be created.
            nx (np.ndarray, optional): The x-component of the surface normals.
                Defaults to None.
            ny (np.ndarray, optional): The y-component of the surface normals.
                Defaults to None.
            nz (np.ndarray, optional): The z-component of the surface normals.
                Defaults to None.
        """
        return InteractionParams()

    def _compute_aoi(self, rays, nx, ny, nz):
        """
        Computes the angle of incidence for the given rays and surface normals.

        Args:
            rays: The rays.
            nx: The x-component of the surface normals.
            ny: The y-component of the surface normals.
            nz: The z-component of the surface normals.

        Returns:
            np.ndarray: The angle of incidence for each ray.
        """
        dot = np.abs(nx * rays.L + ny * rays.M + nz * rays.N)
        dot = np.clip(dot, -1, 1)  # required due to numerical precision
        return np.arccos(dot)    

    @abstractmethod
    def reflect(self, params: InteractionParams):
        """
        Abstract method to handle reflection interaction.

        Args:
            params (InteractionParams): The parameters for the interaction.

        Returns:
            rays (RealRays): The rays after the interaction.
        """
        return params.rays

    @abstractmethod
    def transmit(self, params: InteractionParams):
        """
        Abstract method to handle transmission interaction.

        Args:
            params (InteractionParams): The parameters for the interaction.

        Returns:
            rays (RealRays): The rays after the interaction.
        """
        return params.rays


class SimpleCoating(BaseCoating):
    """
    A simple coating class that represents a coating with given transmittance
    and reflectance.

    Args:
        transmittance (float): The transmittance of the coating.
        reflectance (float, optional): The reflectance of the coating.
            Defaults to 0.

    Attributes:
        transmittance (float): The transmittance of the coating.
        reflectance (float): The reflectance of the coating.
        absorptance (float): The absorptance of the coating, calculated
            as 1 - reflectance - transmittance.

    Methods:
        reflect(params: InteractionParams) -> Rays:
            Reflects the rays based on the reflectance of the coating.
        transmit(params: InteractionParams) -> Rays:
            Transmits the rays based on the transmittance of the coating.
    """

    def __init__(self, transmittance, reflectance=0):
        self.transmittance = transmittance
        self.reflectance = reflectance
        self.absorptance = 1 - reflectance - transmittance

    def reflect(self, params: InteractionParams):
        """
        Reflects the rays based on the reflectance of the coating.

        Args:
            params (InteractionParams): The parameters for the interaction.

        Returns:
            rays (RealRays): The rays after reflection.
        """
        rays = params.rays
        rays.e *= self.reflectance
        return rays

    def transmit(self, params: InteractionParams):
        """
        Transmits the rays through the coating by multiplying their energy
        with the transmittance.

        Args:
            params (InteractionParams): The parameters for the interaction.

        Returns:
            rays (RealRays): The rays after transmission.
        """
        rays = params.rays
        rays.e *= self.transmittance
        return rays


class PolarizedCoating(BaseCoating):

    def create_interaction_params(self, rays, nx, ny, nz):
        """
        Create interaction parameters for the given rays.

        Args:
            rays (RealRays): A list of rays for which interaction parameters
                need to be created.
        """
        L0 = np.copy(np.atleast_1d(rays.L))
        M0 = np.copy(np.atleast_1d(rays.M))
        N0 = np.copy(np.atleast_1d(rays.N))
        aoi = self._compute_aoi(rays, nx, ny, nz)
        return InteractionParams(L0=L0, M0=M0, N0=N0, aoi=aoi)

    def reflect(self, params: InteractionParams):
        jones_matrix = self._jones_matrix(params, reflect=True)
        self._update_polarization_matrices(params, jones_matrix)
        return params.rays

    def transmit(self, params: InteractionParams):
        jones_matrix = self._jones_matrix(params, reflect=False)
        self._update_polarization_matrices(params, jones_matrix)
        return params.rays

    def _update_polarization_matrices(self, params: InteractionParams,
                                      jones_matrix: np.ndarray):
        # define local variables
        rays = params.rays
        L0 = params.L0
        M0 = params.M0
        N0 = params.N0

        # merge k-vector components into matrix for speed
        k0 = np.array([L0, M0, N0]).T
        k1 = np.array([rays.L, rays.M, rays.N]).T

        # find s-component
        s = np.cross(k0, k1)
        mag = np.linalg.norm(s, axis=1)

        # handle case when mag = 0 (i.e., k0 parallel to k1)
        if np.any(mag == 0):
            s[mag == 0] = np.cross(k0[mag == 0], np.array([1, 1e-10, 0]))
            mag = np.linalg.norm(s, axis=1)

        s /= mag[:, np.newaxis]

        # find p-component pre and post surface
        p0 = np.cross(k0, s)
        p1 = np.cross(k1, s)

        # othogonal transformation matrices
        o_in = np.stack((s, p0, k0), axis=1)
        o_out = np.stack((s, p1, k1), axis=2)

        # compute polarization matrix for surface
        p = np.einsum('nij,njk,nkl->nil', o_out, jones_matrix, o_in)

        # update polarization matrices of rays
        rays.p = np.matmul(p, rays.p)

        # singular values of p represent rs and rp transmission on this surface
        singular_values = np.linalg.svd(rays.p, compute_uv=False)

        # update ray energies
        rays.e = (np.abs(singular_values[:, 1])**2 +
                  np.abs(singular_values[:, 2])**2)

    def _jones_matrix(self, params: InteractionParams, reflect: bool = False):
        return np.tile(np.eye(3), (params.rays.x.size, 1, 1))


class FresnelCoating(PolarizedCoating):

    def __init__(self, material_pre, material_post):
        self.material_pre = material_pre
        self.material_post = material_post

    def _jones_matrix(self, params: InteractionParams, reflect: bool = False):
        # define local variables
        rays = params.rays
        aoi = params.aoi
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)

        # precompute cosines for speed
        cos_theta_i = np.cos(aoi)
        cos_theta_t = np.sqrt(1 - (n1 / n2 * np.sin(aoi))**2)

        # compute fresnel coefficients
        if reflect:
            s = (n1 * cos_theta_i - n2 * cos_theta_t) / \
                (n1 * cos_theta_i + n2 * cos_theta_t)
            p = (n2 * cos_theta_i - n1 * cos_theta_t) / \
                (n2 * cos_theta_i + n1 * cos_theta_t)
        else:
            s = 2 * n1 * cos_theta_i / (n1 * cos_theta_i + n2 * cos_theta_t)
            p = 2 * n1 * cos_theta_i / (n2 * cos_theta_i + n1 * cos_theta_t)

        # create jones matrix
        jones_matrix = np.zeros((rays.x.size, 3, 3))
        jones_matrix[:, 0, 0] = s
        jones_matrix[:, 1, 1] = p
        jones_matrix[:, 2, 2] = 1

        return jones_matrix
