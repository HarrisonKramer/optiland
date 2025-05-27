"""Huygens-Fresnel Point Spread Function (PSF) Module

This module provides functionality for simulating and analyzing the Point
Spread Function (PSF) of optical systems using the Huygens-Fresnel principle.
It includes capabilities for generating PSF for a given optical system.
Visualization and Strehl ratio calculation are primarily handled by the base
class. The PSF is normalized against the peak of an ideal diffraction-limited
system calculated using the same Huygens-Fresnel principle.

Kramer Harrison, 2025
"""

from numba import njit, prange

import optiland.backend as be
from optiland.psf.base import BasePSF
from optiland.visualization.utils import transform
from optiland.wavefront import Wavefront


class HuygensPSF(BasePSF):
    """Huygens PSF class using Huygens-Fresnel principle.

    Computes PSF by integrating contributions from point sources across the
    exit pupil. The PSF is normalized such that the peak intensity of an
    equivalent diffraction-limited system (same pupil amplitude, zero phase error)
    would be 100.0. This makes the Strehl ratio (peak of actual PSF / 100.0,
    or value at PSF center / 100.0) meaningful.

    Inherits from `BasePSF` for common initialization (Wavefront setup) and
    visualization methods.

    Args:
        optic (Optic): The optical system object, containing properties like
            paraxial data and surface information.
        field (tuple): The field point (e.g., (Hx, Hy) in normalized field
            coordinates) at which to compute the PSF.
        wavelength (float): The wavelength of light in micrometers.
        num_rays (int, optional): The number of rays used to sample the pupil
            plane along one dimension. The pupil will be a grid of
            `num_rays` x `num_rays`. Defaults to 128.
        image_size (int, optional): The size of the image grid for PSF
            calculation. Defaults to 128.
    """

    def __init__(self, optic, field, wavelength, num_rays=128, image_size=128):
        if be.get_backend() != "numpy":
            raise ValueError("HuygensPSF only supports numpy backend.")

        super().__init__(
            optic=optic, field=field, wavelength=wavelength, num_rays=num_rays
        )

        self.cx = None  # center of the image plane
        self.cy = None
        self.pixel_pitch = None  # pixel pitch of image plane in m

        self.image_size = image_size
        self.psf = self._compute_psf()

    def _get_image_extent(self):
        """Calculate the extent of the image plane based on the optic's parameters.

        This method computes the extent of the image plane based on the geometric
        spot size, as well as a scaled ideal Airy disk at a given wavelength. The
        extent is defined as the maximum of the geometric extent and the ideal
        extent, ensuring that the PSF covers the area where the light is expected
        to be distributed.
        """
        Hx, Hy = self.fields[0]  # single field point
        rays = self.optic.trace(
            Hx=Hx,
            Hy=Hy,
            wavelength=self.wavelengths[0],
            distribution="hexapolar",
            num_rays=6,
        )
        rx, ry, rz = transform(
            rays.x, rays.y, rays.z, self.optic.image_surface, is_global=True
        )
        self.cx = be.mean(rx)
        self.cy = be.mean(ry)

        num_Airy_disks = 5.0  # how many Airy disk radii to include in half-extent
        extent_geometric = be.max(be.hypot(rx - self.cx, ry - self.cy))
        extent_ideal = (
            num_Airy_disks
            * self._get_effective_FNO()  # effective F-number
            * 1.22
            * (self.wavelengths[0] * 1e-3)  # um --> mm
        )

        extent = max(extent_geometric, extent_ideal)

        # Calculate pixel pitch
        self.pixel_pitch = 2 * extent / self.image_size

        xmin = -extent + self.cx
        xmax = extent + self.cx
        ymin = -extent + self.cy
        ymax = extent + self.cy

        return xmin, xmax, ymin, ymax

    def _get_image_coordinates(self):
        """Generate image coordinates for the PSF calculation.

        This method computes the coordinates of the image plane based on
        the extent of the image surface. It creates a grid of points in the
        image plane, which will be used to evaluate the PSF using the
        Huygens-Fresnel principle.
        """
        xmin, xmax, ymin, ymax = self._get_image_extent()
        image_x = be.linspace(xmin, xmax, self.image_size)
        image_y = be.linspace(ymin, ymax, self.image_size)
        image_x, image_y = be.meshgrid(image_x, image_y)

        img_geometry = self.optic.image_surface.geometry
        image_z = img_geometry.sag(image_x, image_y)

        # transform image coordinates from local to global frame
        image_x, image_y, image_z = transform(
            image_x, image_y, image_z, self.optic.image_surface, is_global=False
        )

        image_x = be.reshape(image_x, (self.image_size, self.image_size))
        image_y = be.reshape(image_y, (self.image_size, self.image_size))
        image_z = be.reshape(image_z, (self.image_size, self.image_size))

        return image_x, image_y, image_z

    def _get_normalization(self, pupil_data):
        pass

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _huygens_fresnel_summation(
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
            image_x, image_y, image_z (np.ndarray): 2D arrays of image surface coords.
            pupil_x, pupil_y, pupil_z (np.ndarray): 1D arrays of pupil plane coords.
            pupil_amp (np.ndarray): 1D array of pupil plane amplitudes.
            pupil_opd (np.ndarray): 1D array of optical path difference in mm.
            wavelength (float): Wavelength of the light in mm.
            Rp (float): Radius of the exit pupil reference sphere in mm.

        Returns:
            np.ndarray: 2D array of the point spread function.
        """
        k = 2.0 * be.pi / wavelength  # wavenumber
        Nx, Ny = image_x.shape
        field = be.zeros((Nx, Ny), dtype=be.complex128)

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
                    R = be.sqrt(dx * dx + dy * dy + dz * dz)

                    # Spherical propagation kernel
                    wave = be.exp(1j * k * R) / R

                    # Compute the unit normal at the pupil point
                    nux = u / Rp
                    nuy = v / Rp
                    nuz = w / Rp

                    # Compute the cosine of the angle between (P - Q) and pupil normal.
                    # P is the image point, Q is the pupil point
                    dot = dx * nux + dy * nuy + dz * nuz
                    cos_theta = dot / R
                    Q_obliq = 0.5 * (1.0 + cos_theta)  # obliquity factor

                    # Pupil function
                    pupil_phase = be.exp(1j * k * pupil_opd[j])

                    # Add contribution to the field sum
                    sum_val += pupil_amp[j] * pupil_phase * wave * Q_obliq

                field[ix, iy] = sum_val

        # Compute psf as the squared magnitude of the field
        psf = be.abs(field) ** 2
        return psf

    def _get_normalization(self):
        """Calculates the normalization factor for the PSF.

        This factor ensures that an ideal, diffraction-limited system (no
        aberrations, uniform unit amplitude transmission across the pupil)
        would have a peak PSF intensity corresponding to a Strehl ratio of 1.0
        (or 100% when scaled by `_compute_psf`).

        The normalization is based on the peak intensity of a PSF computed
        from an idealized pupil: one with uniform amplitude (1.0) and zero
        phase within the aperture defined by the actual system's first pupil,
        and zero outside.

        Returns:
            float: The normalization factor.
        """
        if self.fields[0] == (0, 0):
            data = self.get_data((0, 0), self.wavelengths[0])
        else:
            wf = Wavefront(
                self.optic,
                distribution="uniform",
                num_rays=self.num_rays,
                fields=((0, 0),),
                wavelengths=(self.wavelengths[0],),
            )
            data = wf.get_data((0, 0), self.wavelengths[0])

        pupil_opd_ideal = be.zeros_like(data.opd)  # ideal case has no OPD
        image_x = be.zeros((1, 1))  # single point for normalization
        image_y = be.zeros((1, 1))
        ideal_z = self.optic.surface_group.positions[-1]  # image plane position
        image_z = be.full((1, 1), ideal_z)

        psf_max = self._huygens_fresnel_summation(
            image_x,
            image_y,
            image_z,
            data.pupil_x,
            data.pupil_y,
            data.pupil_z,
            data.intensity,
            pupil_opd_ideal,
            self.wavelengths[0] * 1e-3,
            data.radius.item(),
        )

        return psf_max[0, 0]  # Normalize by the peak of the ideal PSF

    def _compute_psf(self):
        """Compute the PSF using the Huygens-Fresnel principle."""
        # Retrieve pupil data from Wavefront instance
        Hx, Hy = self.fields[0]
        wavelength_um = self.wavelengths[0]
        wavelength_mm = wavelength_um * 1e-3
        data = self.get_data((Hx, Hy), wavelength_um)

        # Extract pupil data
        pupil_x, pupil_y, pupil_z = data.pupil_x, data.pupil_y, data.pupil_z
        pupil_amp = data.intensity
        pupil_opd = data.opd * wavelength_mm  # waves to mm
        Rp = data.radius.item()  # Radius of curvature of exit pupil

        # Get image coordinates
        image_x, image_y, image_z = self._get_image_coordinates()

        # Compute the PSF using Huygens-Fresnel summation
        psf = self._huygens_fresnel_summation(
            image_x,
            image_y,
            image_z,
            pupil_x,
            pupil_y,
            pupil_z,
            pupil_amp,
            pupil_opd,
            wavelength_mm,
            Rp,
        )

        # Normalize the PSF
        normalization = self._get_normalization()
        psf = psf / normalization * 100.0

        return psf

    def _get_psf_units(self, image):
        """Calculates the physical extent (units) of the PSF image for plotting.

        This method is called by `BasePSF.view()` to determine axis labels.
        It computes the total spatial width and height (in micrometers) of the
        provided PSF image data.

        Args:
            image (be.ndarray): The PSF image data (often a
                zoomed/cropped version from `BasePSF.view`). Its shape is used
                to determine the total extent for labeling.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the physical
            total width and total height of the PSF image area, in micrometers.
            These are returned as NumPy arrays as `BasePSF.view` expects them
            for Matplotlib's `extent` argument.
        """
        num_x, num_y = image.shape
        dx = self.pixel_pitch
        x = be.to_numpy(num_x * dx) * 1e3  # mm --> um
        y = be.to_numpy(num_y * dx) * 1e3  # mm --> um
        return x, y
