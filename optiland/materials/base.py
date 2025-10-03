"""Basae Material

This module defines the base class for materials. The base class provides
methods to calculate the refractive index, extinction coefficient, and Abbe
number of a material. Subclasses of BaseMaterial should implement the `n` and
`k` methods to provide specific material properties.

Kramer Harrison, 2024
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import optiland.backend as be
from optiland.environment.manager import environment_manager


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
        n(wavelength: float | be.ndarray) -> float | be.ndarray:
            Abstract method to calculate the refractive index at a given
            wavelength(s) in microns.
        k(wavelength: float | be.ndarray) -> float | be.ndarray:
            Abstract method to calculate the extinction coefficient at a given
            wavelength(s) in microns.
        abbe() -> float:
            Method to calculate the Abbe number of the material.

    """

    _registry = {}

    def __init__(self):
        """Initializes the material and its caches."""
        self._n_cache = {}
        self._k_cache = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseMaterial._registry[cls.__name__] = cls

    def _create_cache_key(self, wavelength: float | be.ndarray, **kwargs) -> tuple:
        """Creates a hashable cache key from wavelength and kwargs."""
        if be.is_array_like(wavelength):
            wavelength_key = tuple(np.ravel(be.to_numpy(wavelength)))
        else:
            wavelength_key = wavelength
        return (wavelength_key,) + tuple(sorted(kwargs.items()))

    def n(self, wavelength: float | be.ndarray, **kwargs) -> float | be.ndarray:
        """
        Calculates the refractive index relative to the current environment medium.

        This method ensures that all index calculations are consistent with the
        globally defined immersion medium, providing perfect backwards
        compatibility when the environment is standard air.
        """
        cache_key = self._create_cache_key(wavelength, **kwargs)
        if cache_key in self._n_cache:
            return self._n_cache[cache_key]

        # 1. Get the absolute index of this material
        n_absolute_self = self._calculate_absolute_n(wavelength, **kwargs)

        # 2. Get the absolute index of the environment medium
        env_medium = environment_manager.get_environment().medium
        n_absolute_env = env_medium._calculate_absolute_n(wavelength, **kwargs)

        # 3. Return the relative index
        result = n_absolute_self / n_absolute_env
        self._n_cache[cache_key] = result
        return result

    def k(self, wavelength: float | be.ndarray, **kwargs) -> float | be.ndarray:
        """Calculates the extinction coefficient at a given wavelength with caching.

        Args:
            wavelength (float | be.ndarray): The wavelength(s) of light in microns.
                Can be a float, numpy array, or torch tensor.
            **kwargs: Additional keyword arguments for calculation.

        Returns:
            float | be.ndarray: The extinction coefficient at the given wavelength(s).
        """
        cache_key = self._create_cache_key(wavelength, **kwargs)

        if cache_key in self._k_cache:
            return self._k_cache[cache_key]

        result = self._calculate_k(wavelength, **kwargs)
        self._k_cache[cache_key] = result
        return result

    @abstractmethod
    def _calculate_absolute_n(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """
        Calculates the absolute refractive index (relative to vacuum).
        Subclasses must implement this method.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _calculate_k(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Calculates the extinction coefficient at a given wavelength.

        Args:
            wavelength (float | be.ndarray): The wavelength(s) of light in microns.

        Returns:
            float | be.ndarray: The extinction coefficient at the given
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
