"""Uniform Apodization Module

This module provides a class for uniform apodization, which applies a constant intensity
scaling factor of 1.0 to all rays.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be

from .base import BaseApodization


class UniformApodization(BaseApodization):
    """Uniform apodization, resulting in intensities of 1.0 for all rays."""

    def get_intensity(self, Px, Py):
        """Applies uniform apodization.

        Args:
            Px (be.ndarray): Normalized x pupil coordinates.
            Py (be.ndarray): Normalized y pupil coordinates.

        Returns:
            be.ndarray: Array of intensity scaling factors (all ones).
        """
        return be.ones_like(Px)

    @classmethod
    def from_dict(cls, data):
        """Creates an instance of UniformApodization from a dictionary.

        Args:
            data (dict): A dictionary representation of the apodization.

        Returns:
            UniformApodization: An instance of the UniformApodization class.
        """
        return cls()
