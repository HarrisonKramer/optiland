"""Base Material

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
from optiland.propagation.base import BasePropagationModel
from optiland.propagation.homogeneous import HomogeneousPropagation


class BaseMaterial(ABC):
    """Base class for materials.

    This class defines the interface for material properties such as
    refractive index (n) and extinction coefficient (k). It also provides a
    method to calculate the Abbe number.

    Subclasses of BaseMaterial should implement the abstract methods `n` and
    `k` to provide specific material properties.

    Attributes:
        propagation_model: The model used to propagate rays through this
            material.

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

    def __init__(self, propagation_model: BasePropagationModel | None = None):
        """Initializes the material and its caches.

        Args:
            propagation_model: The propagation model to use for this material.
                If None, a default HomogeneousPropagation model is created.
        """
        self._n_cache = {}
        self._k_cache = {}

        if propagation_model is None:
            self.propagation_model = HomogeneousPropagation(self)
        else:
            self.propagation_model = propagation_model

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
        """Calculates the refractive index at a given wavelength with caching.

        Args:
            wavelength (float | be.ndarray): The wavelength(s) of light in microns.
                Can be a float, numpy array, or torch tensor.
            **kwargs: Additional keyword arguments for calculation (e.g., temperature).

        Returns:
            float | be.ndarray: The refractive index at the given wavelength(s).
        """
        cache_key = self._create_cache_key(wavelength, **kwargs)

        if cache_key in self._n_cache:
            return self._n_cache[cache_key]

        result = self._calculate_n(wavelength, **kwargs)
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
    def _calculate_n(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Calculates the refractive index at a given wavelength.

        Args:
            wavelength (float | be.ndarray): The wavelength(s) of light in microns.

        Returns:
            float | be.ndarray: The refractive index at the given wavelength(s).
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
        return {
            "type": self.__class__.__name__,
            "propagation_model": self.propagation_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        """Create a material from a dictionary representation.

        This factory method first delegates to the appropriate subclass to
        create the material instance, then handles the deserialization of
        the propagation model.

        Args:
            data (dict): The dictionary representation of the material.

        Returns:
            BaseMaterial: An instance of a specific material subclass created
            from the dictionary data.

        """
        material_type = data.get("type")
        if material_type not in cls._registry:
            raise ValueError(f"Unknown material type: {material_type}")

        # Delegate to the correct subclass to create the instance.
        material_subclass = cls._registry[material_type]
        material = material_subclass.from_dict(data)

        # Handle the propagation model deserialization here.
        propagation_model_data = data.get("propagation_model")
        if propagation_model_data:
            # Create the model, passing the material to resolve dependencies.
            new_prop_model = BasePropagationModel.from_dict(
                propagation_model_data, material=material
            )
            # Overwrite the default propagation model.
            material.propagation_model = new_prop_model

        return material
