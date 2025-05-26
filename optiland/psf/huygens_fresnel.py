"""Huygens-Fresnel Point Spread Function (PSF) Module

This module provides functionality for simulating and analyzing the Point
spread function (PSF) of optical systems using the Hugens-Fresnel equation.
It includes capabilities for generating PSF from given wavefront
aberrations, visualizing the PSF in both 2D and 3D projections, and
calculating the Strehl ratio, which is a measure of the quality of an optical
system.

Kramer Harrison, 2025
"""

import numpy as np
from numba import njit, prange

from optiland.psf.fft import FFTPSF
from optiland.visualization.utils import transform
from optiland.wavefront import Wavefront


@njit(parallel=True, fastmath=True)
def compute_psf(
    image_x,
    image_y,
    image_z,
    pupil_x,
    pupil_y,
    pupil_z,
    pupil_amp,
    pupil_opd,
    wavelength,
    Rp,
):
    """
    Compute the point spread function using the Huygens–Fresnel diffraction integral

    Args:
        image_x, image_y, image_z (np.ndarray): 2D arrays of image surface coordinates.
        pupil_x, pupil_y, pupil_z (np.ndarray): 1D arrays of pupil plane coordinates.
        pupil_amp (np.ndarray): 1D array of pupil plane amplitudes.
        pupil_opd (np.ndarray): 1D array of optical path difference.
        z_img (float): Image plane distance from the pupil plane.
        wavelength (float): Wavelength of the light.
        Rp (float): Radius of the pupil plane.

    Returns:
        np.ndarray: 2D array of the point spread function.
    """
    k = 2.0 * np.pi / wavelength  # wavenumber
    Nx, Ny = image_x.shape
    field = np.zeros((Nx, Ny), dtype=np.complex128)

    # Constant factor
    const = 1.0 / (1j * wavelength)

    # Loop over all image points
    for ix in prange(Nx):
        for iy in range(Ny):
            x = image_x[ix, iy]
            y = image_y[ix, iy]
            z = image_z[ix, iy]
            sum_val = 0.0 + 0.0j  # initialize field sum for image point (x, y)

            # Loop over all pupil points
            for j in range(pupil_x.shape[0]):
                u = pupil_x[j]
                v = pupil_y[j]
                w = pupil_z[j]

                # Compute distance R from the pupil point to the image point
                dx = x - u
                dy = y - v
                dz = z - w
                R = np.sqrt(dx * dx + dy * dy + dz * dz)

                # Spherical propagation kernel:
                wave = np.exp(1j * k * R) / R

                # Compute the unit normal at the pupil point
                nux = u / Rp
                nuy = v / Rp
                nuz = w / Rp

                # Compute the cosine of the angle between (P - Q) and the pupil normal.
                # P is the image point, Q is the pupil point
                dot = dx * nux + dy * nuy + dz * nuz
                cos_theta = dot / R
                Q_obliq = 0.5 * (1.0 + cos_theta)  # obliquity factor

                # Pupil function
                pupil_phase = np.exp(1j * k * pupil_opd[j])

                # Add contribution to the field sum
                sum_val += pupil_amp[j] * pupil_phase * wave * Q_obliq

            field[ix, iy] = const * sum_val

    # Compute psf as the squared magnitude of the field
    psf = np.abs(field) ** 2
    return psf


class HuygensPSF(FFTPSF):
    """Huygens PSF class

    This class extends the FFTPSF class to implement the Huygens-Fresnel
    equation for simulating the point spread function of optical systems.
    It provides methods for generating PSF from wavefront aberrations,
    visualizing the PSF, and calculating the Strehl ratio.
    """

    def __init__(self, optic, field, wavelength, num_rays=128, image_size=128):
        super().__init__(optic, field, wavelength, num_rays, grid_size=1024)
        self.image_size = image_size

    def _compute_psf(self):
        # extract calculation parameters
        Hx, Hy = self.field
        wavelength = self.wavelength
        lens = self.optic

        # compute wavefront data at exit pupil
        wf = Wavefront(lens, distribution="uniform", num_rays=self.num_rays)
        data = wf.get_data((Hx, Hy), wavelength)

        # extract pupil data
        pupil_x, pupil_y, pupil_z = data.pupil_x, data.pupil_y, data.pupil_z
        pupil_amp = data.intensity
        pupil_opd = data.opd * (wavelength * 1e-3)
        wavelength = wavelength
        Rp = data.radius.item()

        # find the mean image position
        rays = lens.trace(
            Hx=Hx, Hy=Hy, wavelength=wavelength, distribution="hexapolar", num_rays=6
        )
        rx, ry, rz = transform(
            rays.x, rays.y, rays.z, lens.image_surface, is_global=True
        )
        cx = np.mean(rx)
        cy = np.mean(ry)

        N = self.image_size
        # TODO: update for case that geometric extent is less than diffraction limit
        # --> use entrance pupil diameter, lens.paraxial.EPD()
        extent = np.max(np.hypot(rx - cx, ry - cy))
        image_x = np.linspace(-extent / 1e3 + cx, extent / 1e3 + cx, N)
        image_y = np.linspace(-extent / 1e3 + cy, extent / 1e3 + cy, N)
        image_x, image_y = np.meshgrid(image_x, image_y)
        img_geometry = lens.image_surface.geometry
        image_z = img_geometry.sag(image_x, image_y)

        image_x, image_y, image_z = transform(
            image_x, image_y, image_z, lens.image_surface, is_global=False
        )
        image_x = image_x.reshape(N, N)
        image_y = image_y.reshape(N, N)
        image_z = image_z.reshape(N, N)

        return compute_psf(
            image_x,
            image_y,
            image_z,
            pupil_x,
            pupil_y,
            pupil_z,
            pupil_amp,
            pupil_opd,
            wavelength * 1e-3,
            Rp,
        )
