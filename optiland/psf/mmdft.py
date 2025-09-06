from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.psf.fft import FFTPSF, calculate_grid_size

class MMDFTPSF(FFTPSF):
    """Class representing the Matrix Multiply Discrete Fourier Transform (MMDFT) PSF.

    This class computes the PSF of an optical system by taking the Fourier
    Transform of the pupil function. It inherits common visualization and
    initialization functionalities from `BasePSF`.

    If no grid size is specified, OpticStudio's FFT PSF sampling behavior is
    emulated by scaling down the number of rays in the pupil and using a
    grid size of `num_rays * 2`.

    Args:
        optic (Optic): The optical system object, containing properties like
            paraxial data and surface information.
        field (tuple): The field point (e.g., (Hx, Hy) in normalized field
            coordinates) at which to compute the PSF.
        wavelength (float): The wavelength of light in micrometers.
        num_rays (int, optional): The number of rays used to sample the pupil
            plane along one dimension. The pupil will be a grid of
            `num_rays` x `num_rays`. Defaults to 128.
        grid_size (int, optional): The size of the grid used for FFT
            computation (includes zero-padding). This determines the
            resolution of the PSF. Defaults to 1024. If not specified,
            it is calculated based on `num_rays`.
        strategy (str): The calculation strategy to use. Supported options are
            "chief_ray", "centroid_sphere", and "best_fit_sphere".
            Defaults to "chief_ray".
        remove_tilt (bool): If True, removes tilt and piston from the OPD data.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the strategy.

    Attributes:
        pupils (list[be.ndarray]): A list of complex-valued pupil functions,
            one for each wavelength used in the `Wavefront` parent. Each pupil
            is a 2D array.
        psf (be.ndarray): The computed Point Spread Function. This is a 2D
            array representing the intensity distribution in the image plane,
            normalized such that a diffraction-limited system has a peak of 100.
        grid_size (int): The size of the grid used for FFT computation.
        num_rays (int): The number of rays used to sample the pupil. This is
                        the `num_rays` passed during initialization and used
                        by `Wavefront` for generating OPD/intensity data.
    """

    # TODO: Documentation
    def __init__(
        self,
        optic,
        field,
        wavelength,
        num_rays=128,
        grid_size=None,
        pixel_pitch=None,
        strategy="chief_ray",
        remove_tilt=False,
        **kwargs,
    ):
        if grid_size is None:
            if num_rays < 32:
                raise ValueError(
                    "num_rays must be at least 32 if grid_size is not specified."
                )
            num_rays, grid_size = calculate_grid_size(num_rays)

        super().__init__(
            optic=optic,
            field=field,
            wavelength=wavelength,
            num_rays=num_rays,
            strategy=strategy,
            remove_tilt=remove_tilt,
            **kwargs,
        )

        if pixel_pitch is None:
            pixel_pitch = be.min(self.wavelengths) * self._get_working_FNO() / 2

        self.grid_size = grid_size
        self.pixel_pitch = pixel_pitch
        self.pupils = self._generate_pupils()
        self.psf = self._compute_psf()

    def _compute_psf(self):
        #TODO: Documentation
        if not self.pupils:
            raise ValueError(
                "Pupil functions have not been generated prior to _compute_psf call."
            )

        left_kernels, right_kernels = self._compute_base_kernels()
        psf = []
        for p, lk, rk in zip(self.pupils, left_kernels, right_kernels):
            image_plane = be.matmul(lk, be.matmul(p, rk))
            psf.append(image_plane * be.conj(image_plane))
        psf = be.stack(psf)

        # TODO: Normalization??
        return be.real(be.sum(psf, axis=0))

    # TODO: Strehl Ratio Calculations (ties in with normalization)

    def _compute_base_kernels(self):
        # TODO: Documentation
        # clear_size is a reference to the number of pixels in the pupil grid
        # covering the pupil. The assumption here is that num_rays spans the X and Y
        # extend of the pupil with no pixels inside having an amplitude of zero.
        clear_size = self.num_rays
        Qs = self.wavelengths * self._get_working_FNO() / self.pixel_pitch
        pad_sizes = Qs * clear_size

        # MMDFT generates 2 kernels:
        # right kernel = exp[-2 pi (u x) / pad_size_x]
        # left kernel = exp[-2 pi (v y) / pad_size_y]
        # We do an outer product to create the (u x) and (v y) products
        # For non-square arrays, use the following:
        # [a, b] = pupil.shape (pupil plane)
        # [c, d] = psf.shape   (image plane)
        # (u x) = np.outer(a, c)
        # (v y) = np.outer(d, b)
        # This ensures that the shape of the kernels is correct
        in_units = be.arange(self.num_rays) - self.num_rays//2
        out_units = be.arange(self.grid_size) - self.grid_size//2

        right_vector_product = be.outer(in_units, out_units)
        left_vector_product = be.outer(out_units, in_units)

        right_kernels = []
        left_kernels = []

        for pad_size in pad_sizes:
            right_kernel = be.to_complex(
                be.exp(-2j * np.pi * right_vector_product / pad_size)
            )
            right_kernel = right_kernel / be.sqrt(pad_size)
            right_kernels.append(right_kernel)
            left_kernel = be.to_complex(
                be.exp(-2j * np.pi * left_vector_product / pad_size)
            )
            left_kernel = left_kernel / be.sqrt(pad_size)
            left_kernels.append(left_kernel)
        return left_kernels, right_kernels

# QUESTIONS FOR THE GROUP:
# 1) Are we worried about spectral weightings at all?
# 2) When we calculate a polychromatic PSF, do we consider the normalization factor
# for each wavelength separately, or only compared to the primary wavelength?
# 3) What do we do in polychromatic calculations about the 1/lambda*F factor?
# 4) Is an option for normalization simply energy_in = energy_out?
#    Ex: For a Q=2 or greater PSF, sum(abs(pupil)**2) = sum(psf) over all pixels in
#    pad_size. Can we then use sum(abs(pupil)**2) as a norm factor?
# 5) Does all this talk of normalization fuck with the Strehl calculation?
# 6) Check assumptions:
#    - The total size of the nonzero aperture is num_rays x num_rays
#    - Working_FNO is a valid way to calculate Q (versus image_FNO - I think this is
#    fine)
# 7) Simplifying kernels/memory - have one single kernel and then modify by raising
#    kernels to power based on wavelength
#    pad_size(lambda) = Q * clear_size = lambda * fno * clear_size / pixel_pitch
#    pad_size(lambda1)/pad_size(lambda2) = lambda1/lambda2
#    k(lambda1) = exp[-2 pi u x / pad_size(lambda1)]
#    k(lambda2) = k(lambda1) ** (lambda1/lambda2)