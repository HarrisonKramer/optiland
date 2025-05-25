"""Huygens-Fresnel Point Spread Function (PSF) Module

This module provides functionality for simulating and analyzing the Point
spread function (PSF) of optical systems using the Hugens-Fresnel equation.
It includes capabilities for generating PSF from given wavefront
aberrations, visualizing the PSF in both 2D and 3D projections, and
calculating the Strehl ratio, which is a measure of the quality of an optical
system.

Kramer Harrison, 2025
"""

from optiland.psf.fft import FFTPSF


class HuygensPSF(FFTPSF):
    """Huygens PSF class

    This class extends the FFTPSF class to implement the Huygens-Fresnel
    equation for simulating the point spread function of optical systems.
    It provides methods for generating PSF from wavefront aberrations,
    visualizing the PSF, and calculating the Strehl ratio.
    """

    def __init__(self, optic, field, wavelength, num_rays=128, grid_size=1024):
        super().__init__(optic, field, wavelength, num_rays, grid_size)
