"""Apodization Base Module

This module defines the abstract base class for apodization functions used in optical
simulations.

Kramer Harrison, 2025
"""

from __future__ import annotations

import abc


class BaseApodization(abc.ABC):
    """Abstract base class for apodization functions."""

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseApodization._registry[cls.__name__] = cls

    @abc.abstractmethod
    def get_intensity(self, Px, Py):
        """Applies the apodization function to the given pupil coordinates.

        Args:
            Px (be.ndarray): Normalized x pupil coordinates.
            Py (be.ndarray): Normalized y pupil coordinates.

        Returns:
            be.ndarray: Array of intensity scaling factors.
        """
        pass  # pragma: no cover

    def to_dict(self):
        """Converts the apodization to a dictionary.

        Returns:
            dict: A dictionary representation of the apodization.
        """
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data):
        """Creates an instance of the apodization from a dictionary.

        Args:
            data (dict): A dictionary representation of the apodization.

        Returns:
            BaseApodization: An instance of the apodization class.
        """
        apodization_type = data.get("type")
        if apodization_type not in cls._registry:
            raise ValueError(f"Unknown apodization type: {apodization_type}")

        # Delegate to the correct subclass's from_dict
        return cls._registry[apodization_type].from_dict(data)
