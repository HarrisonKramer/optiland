"""Basae Material

This module defines the base class for materials. The base class provides
methods to calculate the refractive index, extinction coefficient, and Abbe
number of a material. Subclasses of BaseMaterial should implement the `n` and
`k` methods to provide specific material properties.

Kramer Harrison, 2024
"""

from abc import ABC, abstractmethod


class BaseMaterial(ABC):
    """Base class for materials.

    This class defines the interface for material properties such as
    refractive index (n) and extinction coefficient (k). It also provides a
    method to calculate the Abbe number.

    Subclasses of BaseMaterial should implement the abstract methods `n` and
    `k` to provide specific material properties.

    Attributes:
        None

    Methods:
        n(wavelength: float or be.ndarray) -> float or be.ndarray:
            Abstract method to calculate the refractive index at a given
            wavelength(s) in microns.
        k(wavelength: float or be.ndarray) -> float or be.ndarray:
            Abstract method to calculate the extinction coefficient at a given
            wavelength(s) in microns.
        abbe() -> float:
            Method to calculate the Abbe number of the material.

    """

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseMaterial._registry[cls.__name__] = cls

    @abstractmethod
    def n(self, wavelength: float) -> float:  # Subclasses will handle be.ndarray
        """Calculates the refractive index at a given wavelength.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.

        Returns:
            float or be.ndarray: The refractive index at the given wavelength(s).
        """
        pass  # pragma: no cover

    @abstractmethod
    def k(self, wavelength: float) -> float:  # Subclasses will handle be.ndarray
        """Calculates the extinction coefficient at a given wavelength.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.

        Returns:
            float or be.ndarray: The extinction coefficient at the given
            wavelength(s).
        """
        pass  # pragma: no cover

    def abbe(self) -> float:
        """Calculate the Abbe number (Vd) of the material.

        The Abbe number is a measure of the material's dispersion, defined as
        Vd = (n_d - 1) / (n_F - n_C), where n_d, n_F, and n_C are the
        refractive indices at the Fraunhofer d (587.5618 nm), F (486.1327 nm),
        and C (656.2725 nm) spectral lines, respectively.

        Returns:
            float: The Abbe number of the material.

        """
        nD = self.n(0.5875618)
        nF = self.n(0.4861327)
        nC = self.n(0.6562725)
        return (nD - 1) / (nF - nC)

    def to_dict(self):
        """Convert the material to a dictionary.

        Returns:
            dict: The dictionary representation of the material.

        """
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data):
        """Create a material from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the material.

        Returns:
            BaseMaterial: An instance of a specific material subclass created
            from the dictionary data.

        """
        material_type = data.get("type")
        if material_type not in cls._registry:
            raise ValueError(f"Unknown material type: {material_type}")

        # Delegate to the correct subclass's from_dict
        return cls._registry[material_type].from_dict(data)
