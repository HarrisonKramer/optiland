"""Jones Module

The jones module contains classes for Jones matrices in optics. The module
defines the base class BaseJones, which is an abstract class that defines the
interface for Jones matrices. The module also contains classes for specific
Jones matrices, such as JonesFresnel, JonesPolarizerH, JonesPolarizerV,
JonesPolarizerL45, JonesPolarizerL135, JonesPolarizerRCP, JonesPolarizerLCP,
JonesLinearDiattenuator, JonesLinearRetarder, JonesQuarterWaveRetarder, and
JonesHalfWaveRetarder.

Kramer Harrison, 2024
"""

from abc import ABC, abstractmethod

import optiland.backend as be
from optiland.rays import RealRays


class BaseJones(ABC):
    """Base class for Jones matrices.

    The class defines Jones matrices given ray properties. In the general case,
    the Jones matrix is an Nx3x3 array, where N is the number of rays. The
    array is padded to make it 3x3 to account for 3D ray calculations.
    """

    @abstractmethod
    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            aoi (be.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            be.ndarray: The calculated Jones matrix.

        """
        return be.tile(be.eye(3), (rays.x.size, 1, 1))  # pragma: no cover


class JonesFresnel(BaseJones):
    """Class representing the Jones matrix for Fresnel calculations.

    Args:
        material_pre (Material): Material object representing the
            material before the surface.
        material_post (Material): Material object representing the
            material after the surface.

    """

    def __init__(self, material_pre, material_post):
        self.material_pre = material_pre
        self.material_post = material_post

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            aoi (be.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            be.ndarray: The calculated Jones matrix.

        """
        # define local variables
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)

        # precomputations for speed
        cos_theta_i = be.cos(aoi)
        n = n2 / n1
        radicand = (n**2 - be.sin(aoi) ** 2).astype(complex)
        root = be.sqrt(radicand)

        # compute fresnel coefficients & compute jones matrices
        jones_matrix = be.zeros((rays.x.size, 3, 3), dtype=complex)
        if reflect:
            s = (cos_theta_i - root) / (cos_theta_i + root)
            p = (n**2 * cos_theta_i - root) / (n**2 * cos_theta_i + root)

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


class JonesPolarizerH(BaseJones):
    """Class representing the Jones matrix for a horizontal polarizer."""

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            aoi (be.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            be.ndarray: The calculated Jones matrix.

        """
        jones_matrix = be.zeros((rays.x.size, 3, 3), dtype=complex)
        jones_matrix[:, 0, 0] = 1
        jones_matrix[:, 1, 1] = 0
        jones_matrix[:, 2, 2] = 1

        return jones_matrix


class JonesPolarizerV(BaseJones):
    """Class representing the Jones matrix for a vertical polarizer."""

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            aoi (be.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            be.ndarray: The calculated Jones matrix.

        """
        jones_matrix = be.zeros((rays.x.size, 3, 3), dtype=complex)
        jones_matrix[:, 0, 0] = 0
        jones_matrix[:, 1, 1] = 1
        jones_matrix[:, 2, 2] = 1

        return jones_matrix


class JonesPolarizerL45(BaseJones):
    """Class representing the Jones matrix for a linear polarizer at 45 degrees."""

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            aoi (be.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            be.ndarray: The calculated Jones matrix.

        """
        jones_matrix = be.zeros((rays.x.size, 3, 3), dtype=complex)
        jones_matrix[:, 0, 0] = 0.5
        jones_matrix[:, 0, 1] = 0.5
        jones_matrix[:, 1, 0] = 0.5
        jones_matrix[:, 1, 1] = 0.5
        jones_matrix[:, 2, 2] = 1

        return jones_matrix


class JonesPolarizerL135(BaseJones):
    """Class representing the Jones matrix for a linear polarizer at 135 degrees."""

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            aoi (be.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            be.ndarray: The calculated Jones matrix.

        """
        jones_matrix = be.zeros((rays.x.size, 3, 3), dtype=complex)
        jones_matrix[:, 0, 0] = 0.5
        jones_matrix[:, 0, 1] = -0.5
        jones_matrix[:, 1, 0] = -0.5
        jones_matrix[:, 1, 1] = 0.5
        jones_matrix[:, 2, 2] = 1

        return jones_matrix


class JonesPolarizerRCP(BaseJones):
    """Class representing the Jones matrix for a right circular polarizer."""

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            aoi (be.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            be.ndarray: The calculated Jones matrix.

        """
        jones_matrix = be.zeros((rays.x.size, 3, 3), dtype=complex)
        jones_matrix[:, 0, 0] = 0.5
        jones_matrix[:, 0, 1] = 1j * 0.5
        jones_matrix[:, 1, 0] = -1j * 0.5
        jones_matrix[:, 1, 1] = 0.5
        jones_matrix[:, 2, 2] = 1

        return jones_matrix


class JonesPolarizerLCP(BaseJones):
    """Class representing the Jones matrix for a left circular polarizer."""

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            aoi (be.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            be.ndarray: The calculated Jones matrix.

        """
        jones_matrix = be.zeros((rays.x.size, 3, 3), dtype=complex)
        jones_matrix[:, 0, 0] = 0.5
        jones_matrix[:, 0, 1] = -1j * 0.5
        jones_matrix[:, 1, 0] = 1j * 0.5
        jones_matrix[:, 1, 1] = 0.5
        jones_matrix[:, 2, 2] = 1

        return jones_matrix


class JonesLinearDiattenuator(BaseJones):
    """Represents a linear diattenuator in Jones calculus.

    Attributes:
        t_min (float): Minimum amplitude transmission coefficient.
        t_max (float): Maximum amplitude transmission coefficient.
        theta (float): Angle of the diattenuator.

    Note:
        The intensity transmission is given by the square of the amplitude
        coefficients.

    Methods:
        calculate_matrix: Calculate the Jones matrix for the given rays.

    """

    def __init__(self, t_min, t_max, theta):
        self.t_min = t_min
        self.t_max = t_max
        self.theta = theta

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            aoi (be.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            be.ndarray: The calculated Jones matrix.

        """
        j00 = (
            self.t_max * be.cos(self.theta) ** 2 + self.t_min * be.sin(self.theta) ** 2
        )
        j0x = self.t_max - self.t_min * be.cos(self.theta) * be.sin(self.theta)
        j11 = (
            self.t_max * be.sin(self.theta) ** 2 + self.t_min * be.cos(self.theta) ** 2
        )

        jones_matrix = be.zeros((rays.x.size, 3, 3), dtype=complex)
        jones_matrix[:, 0, 0] = j00
        jones_matrix[:, 0, 1] = j0x
        jones_matrix[:, 1, 0] = j0x
        jones_matrix[:, 1, 1] = j11
        jones_matrix[:, 2, 2] = 1

        return jones_matrix


class JonesLinearRetarder(BaseJones):
    """Represents a linear retarder in Jones calculus.

    Attributes:
        retardance (float): Retardance of the retarder, or the absolute value
            of the phase difference between the two components of the electric
            field.
        theta (float): Angle of the retarder, i.e., the fast axis orientation.

    Methods:
        calculate_matrix: Calculate the Jones matrix for the given rays.

    """

    def __init__(self, retardance, theta):
        self.retardance = retardance
        self.theta = theta

    def calculate_matrix(
        self,
        rays: RealRays,
        reflect: bool = False,
        aoi: be.ndarray = None,
    ):
        """Calculate the Jones matrix for the given rays.

        Args:
            rays (RealRays): Object representing the rays.
            reflect (bool, optional): Indicates whether the rays are reflected
                or not. Defaults to False.
            aoi (be.ndarray, optional): Array representing the angle of
                incidence. Defaults to None.

        Returns:
            be.ndarray: The calculated Jones matrix.

        """
        d = self.retardance
        t = self.theta
        j00 = be.exp(-1j * d / 2) * be.cos(t) ** 2 + be.exp(1j * d / 2) * be.sin(t) ** 2
        j0x = -1j * be.sin(d / 2) * be.sin(2 * t)
        j11 = be.exp(1j * d / 2) * be.cos(t) ** 2 + be.exp(-1j * d / 2) * be.sin(t) ** 2

        jones_matrix = be.zeros((rays.x.size, 3, 3), dtype=complex)
        jones_matrix[:, 0, 0] = j00
        jones_matrix[:, 0, 1] = j0x
        jones_matrix[:, 1, 0] = j0x
        jones_matrix[:, 1, 1] = j11
        jones_matrix[:, 2, 2] = 1

        return jones_matrix


class JonesQuarterWaveRetarder(JonesLinearRetarder):
    """Represents a quarter-wave retarder in Jones calculus.

    Attributes:
        theta (float): Angle of the retarder, i.e., the fast axis orientation.
            Defaults to 0.

    Methods:
        calculate_matrix: Calculate the Jones matrix for the given rays.

    """

    def __init__(self, theta=0):
        super().__init__(be.pi / 2, theta)


class JonesHalfWaveRetarder(JonesLinearRetarder):
    """Represents a half-wave retarder in Jones calculus.

    Attributes:
        theta (float): Angle of the retarder, i.e., the fast axis orientation.
            Defaults to 0.

    Methods:
        calculate_matrix: Calculate the Jones matrix for the given rays.

    """

    def __init__(self, theta=0):
        super().__init__(be.pi, theta)
