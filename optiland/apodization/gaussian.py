"""Gaussian Apodization Module

This module provides a class for Gaussian apodization, which applies a Gaussian profile
to scale the intensities of rays based on their position in the pupil.

Kramer Harrison, 2025
"""

import optiland.backend as be

from .base import BaseApodization


class GaussianApodization(BaseApodization):
    """Gaussian apodization, scaling intensities based on a Gaussian profile."""

    def __init__(self, sigma: float = 1.0):
        """Initializes GaussianApodization.

        Args:
            sigma (float): Standard deviation of the Gaussian function.
                           Controls the width of the apodization.
        """
        if sigma <= 0:
            raise ValueError("Sigma must be positive for GaussianApodization.")
        self.sigma = sigma

    def apply(self, Px, Py):
        """Applies Gaussian apodization.

        Args:
            Px (be.ndarray): Normalized x pupil coordinates.
            Py (be.ndarray): Normalized y pupil coordinates.

        Returns:
            be.ndarray: Array of intensity scaling factors based on Gaussian profile.
        """
        return be.exp(-(Px**2 + Py**2) / (2 * self.sigma**2))
