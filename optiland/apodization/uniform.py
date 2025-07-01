"""Uniform Apodization Module

This module provides a class for uniform apodization, which applies a constant intensity
scaling factor of 1.0 to all rays.

Kramer Harrison, 2025
"""

import optiland.backend as be

from .base import BaseApodization


class UniformApodization(BaseApodization):
    """Uniform apodization, resulting in intensities of 1.0 for all rays."""

    def apply(self, Px, Py):
        """Applies uniform apodization.

        Args:
            Px (be.ndarray): Normalized x pupil coordinates.
            Py (be.ndarray): Normalized y pupil coordinates.

        Returns:
            be.ndarray: Array of intensity scaling factors (all ones).
        """
        return be.ones_like(Px)
