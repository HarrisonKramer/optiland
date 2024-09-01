from abc import ABC, abstractmethod
import numpy as np
from optiland.rays import RealRays
from optiland.jones import JonesFresnel


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
        dot = np.abs(nx * rays.L0 + ny * rays.M0 + nz * rays.N0)
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
        pass  # pragma: no cover

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
        pass  # pragma: no cover


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
        rays.i *= self.reflectance
        return rays

    def transmit(self, rays: RealRays, nx: np.ndarray = None,
                 ny: np.ndarray = None, nz: np.ndarray = None):
        """
        Transmits the rays through the coating by multiplying their intensity
        with the transmittance.

        Args:
            params (InteractionParams): The parameters for the interaction.

        Returns:
            rays (RealRays): The rays after transmission.
        """
        rays.i *= self.transmittance
        return rays


class BaseCoatingPolarized(BaseCoating, ABC):
    """
    A base class for polarized coatings.

    This class inherits from the `BaseCoating` class and the `ABC`
    (Abstract Base Class) module.

    Methods:
        reflect(rays, nx, ny, nz): Reflects the rays off the coating.
        transmit(rays, nx, ny, nz): Transmits the rays through the coating.
    """
    def reflect(self, rays: RealRays, nx: np.ndarray = None,
                ny: np.ndarray = None, nz: np.ndarray = None):
        """
        Reflects the rays off the coating.

        Args:
            rays (RealRays): The rays to be reflected.
            nx (np.ndarray): The x-component of the surface normal vector.
            ny (np.ndarray): The y-component of the surface normal vector.
            nz (np.ndarray): The z-component of the surface normal vector.

        Returns:
            RealRays: The updated rays after reflection.
        """
        aoi = self._compute_aoi(rays, nx, ny, nz)
        jones = self.jones.calculate_matrix(rays, reflect=True, aoi=aoi)
        rays.update(jones)
        return rays

    def transmit(self, rays: RealRays, nx: np.ndarray = None,
                 ny: np.ndarray = None, nz: np.ndarray = None):
        """
        Transmits the rays through the coating.

        Args:
            rays (RealRays): The rays to be transmitted.
            nx (np.ndarray): The x-component of the surface normal vector.
            ny (np.ndarray): The y-component of the surface normal vector.
            nz (np.ndarray): The z-component of the surface normal vector.

        Returns:
            RealRays: The updated rays after transmission through a surface.
        """
        aoi = self._compute_aoi(rays, nx, ny, nz)
        jones = self.jones.calculate_matrix(rays, reflect=False, aoi=aoi)
        rays.update(jones)
        return rays


class FresnelCoating(BaseCoatingPolarized):
    """
    Represents a Fresnel coating for polarized light.

    This class inherits from the BaseCoatingPolarized class and provides
    interaction functionality for polarized light with uncoated surfaces.
    In general, this updates ray intensities based on the Fresnel equations
    on a surface.

    Attributes:
        material_pre (str): The material before the coating.
        material_post (str): The material after the coating.
        jones (JonesFresnel): The JonesFresnel object, which calculates the
            Jones matrices for given ray properties.
    """

    def __init__(self, material_pre, material_post):
        self.material_pre = material_pre
        self.material_post = material_post

        self.jones = JonesFresnel(material_pre, material_post)
