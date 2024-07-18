from abc import ABC, abstractmethod
import numpy as np
from optiland.rays import RealRays


class BaseCoating(ABC):
    """
    Base class for coatings.

    This class defines the basic structure and behavior of a coating.

    Methods:
        interact: Performs an interaction based on the given parameters.
        reflect: Abstract method to handle reflection interaction.
        transmit: Abstract method to handle transmission interaction.
    """

    def interact(self, rays: RealRays, reflect: bool = False,
                 nx: np.ndarray = None, ny: np.ndarray = None,
                 nz: np.ndarray = None):
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
            return self.reflect(rays, nx, ny, nz)
        else:
            return self.transmit(rays, nx, ny, nz)

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
    def reflect(self, rays: RealRays, nx: np.ndarray = None,
                ny: np.ndarray = None, nz: np.ndarray = None):
        """
        Abstract method to handle reflection interaction.

        Args:
            params (InteractionParams): The parameters for the interaction.

        Returns:
            rays (RealRays): The rays after the interaction.
        """
        return rays

    @abstractmethod
    def transmit(self, rays: RealRays, nx: np.ndarray = None,
                 ny: np.ndarray = None, nz: np.ndarray = None):
        """
        Abstract method to handle transmission interaction.

        Args:
            params (InteractionParams): The parameters for the interaction.

        Returns:
            rays (RealRays): The rays after the interaction.
        """
        return rays


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

    def reflect(self, rays: RealRays, nx: np.ndarray = None,
                ny: np.ndarray = None, nz: np.ndarray = None):
        """
        Reflects the rays based on the reflectance of the coating.

        Args:
            params (InteractionParams): The parameters for the interaction.

        Returns:
            rays (RealRays): The rays after reflection.
        """
        rays.e *= self.reflectance
        return rays

    def transmit(self, rays: RealRays, nx: np.ndarray = None,
                 ny: np.ndarray = None, nz: np.ndarray = None):
        """
        Transmits the rays through the coating by multiplying their energy
        with the transmittance.

        Args:
            params (InteractionParams): The parameters for the interaction.

        Returns:
            rays (RealRays): The rays after transmission.
        """
        rays.e *= self.transmittance
        return rays


class PolarizedCoating(BaseCoating):

    def reflect(self, rays: RealRays, nx: np.ndarray = None,
                ny: np.ndarray = None, nz: np.ndarray = None):
        jones_matrix = self._jones_matrix(rays, reflect=True,
                                          nx=nx, ny=ny, nz=nz)
        rays.update(jones_matrix)
        return rays

    def transmit(self, rays: RealRays, nx: np.ndarray = None,
                 ny: np.ndarray = None, nz: np.ndarray = None):
        jones_matrix = self._jones_matrix(rays, reflect=False,
                                          nx=nx, ny=ny, nz=nz)
        rays.update(jones_matrix)
        return rays

    def _jones_matrix(self, rays: RealRays, reflect: bool = False,
                      nx: np.ndarray = None, ny: np.ndarray = None,
                      nz: np.ndarray = None):
        return np.tile(np.eye(3), (rays.x.size, 1, 1))


class FresnelCoating(PolarizedCoating):

    def __init__(self, material_pre, material_post):
        self.material_pre = material_pre
        self.material_post = material_post

    def _jones_matrix(self, rays: RealRays, reflect: bool = False,
                      nx: np.ndarray = None, ny: np.ndarray = None,
                      nz: np.ndarray = None):
        # define local variables
        aoi = self._compute_aoi(rays, nx, ny, nz)
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)

        # precomputations for speed
        cos_theta_i = np.cos(aoi)
        n = n2 / n1
        radicand = (n**2 - np.sin(aoi)**2).astype(complex)
        root = np.sqrt(radicand)

        # compute fresnel coefficients
        if reflect:
            s = (cos_theta_i - root) / (cos_theta_i + root)
            p = (n**2*cos_theta_i - root) / (n**2*cos_theta_i + root)
        else:
            s = 2 * cos_theta_i / (cos_theta_i + root)
            p = 2 * n * cos_theta_i / (n**2 * cos_theta_i + root)

        # create jones matrix
        jones_matrix = np.zeros((rays.x.size, 3, 3), dtype=complex)
        jones_matrix[:, 0, 0] = s
        jones_matrix[:, 1, 1] = p
        jones_matrix[:, 2, 2] = 1

        return jones_matrix
