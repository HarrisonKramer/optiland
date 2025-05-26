"""FFT Point Spread Function (PSF) Module

This module provides functionality for simulating and analyzing the Point
Spread Function (PSF) of optical systems using the Fast Fourier Transform
(FFT). It includes capabilities for generating PSF from given wavefront
aberrations and calculating the Strehl ratio. Visualization is handled by
the base class.

Kramer Harrison, 2023
"""

import optiland.backend as be
from optiland.psf.base import BasePSF
# No direct numpy, matplotlib, or scipy.ndimage imports needed here anymore.
# BasePSF handles visualization imports.


class FFTPSF(BasePSF):
    """Class representing the Fast Fourier Transform (FFT) PSF.

    This class computes the PSF of an optical system by taking the Fourier
    Transform of the pupil function. It inherits common visualization and
    initialization functionalities from `BasePSF`.

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
            resolution of the PSF. Defaults to 1024.

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

    def __init__(self, optic, field, wavelength, num_rays=128, grid_size=1024):
        """Initializes the FFTPSF object.

        Sets up the wavefront parameters via `BasePSF`, then generates pupil
        functions specific to FFT calculation, and finally computes the PSF.
        """
        super().__init__(
            optic=optic,
            field=field,
            wavelength=wavelength, # Passed to BasePSF -> Wavefront
            num_rays=num_rays,     # Passed to BasePSF -> Wavefront
            grid_size=grid_size,   # Stored in BasePSF, used here
        )
        # self.num_rays is inherited from Wavefront via BasePSF
        # self.grid_size is stored by BasePSF constructor
        self.pupils = self._generate_pupils()
        self.psf = self._compute_psf() # Computes and sets self.psf

    def _generate_pupils(self):
        """Generates complex pupil functions for each wavelength.

        For each specified wavelength, this method:
        1. Obtains wavefront data (Optical Path Difference - OPD, intensity)
           at the exit pupil using the `get_data` method from `Wavefront`.
        2. Creates a 2D grid representing the pupil, sampled with `self.num_rays`
           points across its diameter.
        3. Populates the grid with complex values: `A * exp(j * phi)`, where
           amplitude `A` is derived from ray intensity and phase `phi` from OPD.
           The OPD is converted to phase using the wavelength.

        Returns:
            list[be.ndarray]: A list of complex 2D arrays (shape:
            `num_rays` x `num_rays`), each representing the pupil function
            for a wavelength.
        """
        # Create a spatial grid for the pupil plane. `self.num_rays` (from Wavefront)
        # defines the resolution of this pupil grid.
        coords = be.linspace(-1, 1, self.num_rays) # Normalized coordinates
        x_grid, y_grid = be.meshgrid(coords, coords)
        
        # Flatten grid for easier indexing based on radial distance
        x_flat = be.ravel(x_grid)
        y_flat = be.ravel(y_grid)
        # R_flat is the radial coordinate, used to filter points outside unit circle (pupil edge)
        R_flat = be.sqrt(x_flat**2 + y_flat**2)

        field_point = self.fields[0]  # PSF is for a single field point from BasePSF
        generated_pupils = []

        for wl_micrometers in self.wavelengths: # self.wavelengths from Wavefront (in µm)
            wavefront_data = self.get_data(field_point, wl_micrometers) # From Wavefront

            # Initialize a complex array for the pupil function values
            P_flat = be.zeros_like(x_flat, dtype=be.complex128)

            # OPD is in micrometers. Phase = 2 * pi * OPD / wavelength.
            # wavefront_data.opd is already path length difference.
            phase = (2 * be.pi / wl_micrometers) * wavefront_data.opd
            
            # Amplitude from intensity. Ensure it's normalized if necessary.
            # Current Wavefront seems to provide raw intensity.
            # Normalizing by mean intensity helps stabilize if ray densities vary.
            amplitude = be.sqrt(wavefront_data.intensity) # E-field amplitude is sqrt(Intensity)
            if be.mean(amplitude) > 1e-9: # Avoid division by zero for dark pupils
                amplitude = amplitude / be.mean(amplitude) # Normalize mean amplitude
            else:
                amplitude = be.zeros_like(amplitude)


            # Assign complex values P = amplitude * exp(j * phase) only to points within the unit circle
            # Indices where R_flat <= 1 are inside the pupil
            valid_indices_mask = R_flat <= 1
            
            complex_pupil_values = amplitude * be.exp(1j * phase)
            
            # Apply mask: only fill where valid_indices_mask is true
            P_flat = be.where(valid_indices_mask, complex_pupil_values, P_flat)

            # Reshape the flat array back into a 2D grid (num_rays x num_rays)
            P_grid = be.reshape(P_flat, (self.num_rays, self.num_rays))
            generated_pupils.append(P_grid)

        return generated_pupils

    def _compute_psf(self):
        """Computes the PSF from the generated pupil functions via FFT.

        This involves:
        1. Padding the pupil functions with zeros up to `self.grid_size`.
           This padding determines the PSF's resolution.
        2. Calculating a normalization factor to ensure consistent Strehl ratio
           (diffraction-limited peak = 100%).
        3. For each (padded) pupil:
           a. Performing a 2D FFT and applying `fftshift` to center it.
           b. Calculating the squared magnitude (intensity) of the result.
        4. If multiple wavelengths are used, their individual PSFs are summed
           incoherently (sum of intensities).
        5. Normalizing the final PSF using the pre-calculated factor.

        Returns:
            be.ndarray: The computed 2D PSF (shape: `grid_size` x `grid_size`),
            normalized so that a diffraction-limited system's peak is 100.
        
        Raises:
            ValueError: If pupil functions have not been generated.
        """
        if not self.pupils:
            raise ValueError("Pupil functions have not been generated prior to _compute_psf call.")

        padded_pupils = self._pad_pupils()
        norm_factor = self._get_normalization()

        psf_list_for_summation = []
        for pupil_padded in padded_pupils:
            # The centered Fourier Transform of the pupil function gives the complex amplitude spread function
            amplitude_spread_function = be.fft.fftshift(be.fft.fft2(pupil_padded))
            
            # Intensity PSF is the squared magnitude of the amplitude spread function
            intensity_psf = be.real(amplitude_spread_function * be.conj(amplitude_spread_function))
            psf_list_for_summation.append(intensity_psf)
        
        # Stack PSFs from different wavelengths (if any) along a new axis (axis 0)
        stacked_psfs = be.stack(psf_list_for_summation, axis=0)

        # Incoherent sum for polychromatic PSF: sum intensities from each wavelength
        summed_psf = be.sum(stacked_psfs, axis=0)

        if norm_factor == 0:
            # Avoid division by zero if norm_factor is zero (e.g., fully vignetted or zero-amplitude pupil)
            # Return a zero PSF of the correct shape and type.
            return be.zeros((self.grid_size, self.grid_size), dtype=be.float_dtype())
        
        # Normalize so diffraction-limited peak is 100%
        final_psf = (summed_psf / norm_factor) * 100.0
        return final_psf

    def _pad_pupils(self):
        """Pads the pupil functions with zeros to match `self.grid_size`.

        Zero-padding in the spatial domain (pupil plane) before FFT leads to
        interpolation in the frequency domain (PSF plane). This effectively
        increases the sampling resolution of the computed PSF, showing more detail.

        Returns:
            list[be.ndarray]: A list of padded pupil functions (shape:
            `grid_size` x `grid_size`).
        
        Raises:
            ValueError: If any pupil's dimension (`num_rays`) is larger
                        than `grid_size`.
        """
        padded_pupils_list = []
        for pupil_unpadded in self.pupils:
            if pupil_unpadded.shape[0] > self.grid_size or pupil_unpadded.shape[1] > self.grid_size:
                raise ValueError(
                    f"Pupil size ({pupil_unpadded.shape}) cannot be larger than "
                    f"grid_size ({self.grid_size}). `num_rays` should be <= `grid_size`."
                )

            # Calculate padding amounts for rows (axis 0) and columns (axis 1)
            # This centers the smaller pupil_unpadded array within the larger grid_size array
            pad_before_rows = (self.grid_size - pupil_unpadded.shape[0]) // 2
            pad_after_rows = self.grid_size - pupil_unpadded.shape[0] - pad_before_rows
            
            pad_before_cols = (self.grid_size - pupil_unpadded.shape[1]) // 2
            pad_after_cols = self.grid_size - pupil_unpadded.shape[1] - pad_before_cols
            
            padding_config = (
                (pad_before_rows, pad_after_rows),
                (pad_before_cols, pad_after_cols),
            )

            # Pad with complex zeros (0+0j), as pupils are complex
            padded_pupil = be.pad(
                pupil_unpadded,
                padding_config,
                mode="constant",
                constant_values=(0+0j), 
            )
            padded_pupils_list.append(padded_pupil)
        return padded_pupils_list

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
            float: The normalization factor. Returns 0.0 if `self.pupils` is empty
                   or if the template pupil is entirely zero (e.g. fully vignetted).
        """
        if not self.pupils:
            return 0.0 

        # Use the first pupil in the list as a template for the aperture shape.
        # This assumes all pupils define the same aperture extent, which is typical.
        template_pupil_shape = self.pupils[0].shape
        template_pupil_dtype = self.pupils[0].dtype # Should be complex

        # Create an ideal, unaberrated pupil: amplitude is 1.0 within the aperture, 0.0 outside. Phase is zero.
        # An easy way to define "within the aperture" is to use locations where the
        # actual first pupil `self.pupils[0]` is non-zero.
        # P_ideal has 1.0+0j where self.pupils[0] is transmitting, 0.0+0j otherwise.
        P_ideal_unpadded = be.where(self.pupils[0] != (0+0j), 1.0+0j, 0.0+0j)
        P_ideal_unpadded = be.astype(P_ideal_unpadded, dtype=template_pupil_dtype)


        # Pad this ideal pupil exactly as the actual pupils are padded.
        pad_before_rows = (self.grid_size - template_pupil_shape[0]) // 2
        pad_after_rows = self.grid_size - template_pupil_shape[0] - pad_before_rows
        pad_before_cols = (self.grid_size - template_pupil_shape[1]) // 2
        pad_after_cols = self.grid_size - template_pupil_shape[1] - pad_before_cols
        padding_config = (
            (pad_before_rows, pad_after_rows),
            (pad_before_cols, pad_after_cols),
        )
        P_ideal_padded = be.pad(
            P_ideal_unpadded,
            padding_config,
            mode="constant",
            constant_values=(0+0j),
        )

        # Compute the PSF of this ideal, padded pupil.
        amplitude_spread_function_ideal = be.fft.fftshift(be.fft.fft2(P_ideal_padded))
        psf_ideal_intensity = be.real(amplitude_spread_function_ideal * be.conj(amplitude_spread_function_ideal))
        
        peak_ideal_psf = be.max(psf_ideal_intensity)

        if peak_ideal_psf == 0: # Handle cases like fully vignetted pupil
            return 0.0

        # If multiple wavelengths are summed incoherently in _compute_psf,
        # the normalization factor should reflect the sum of their ideal peaks.
        # Assuming each wavelength contributes equally to an "ideal" polychromatic PSF peak.
        return peak_ideal_psf * len(self.wavelengths)


    def _get_psf_units(self, image_data_for_shape):
        """Calculates the physical extent (units) of the PSF image for plotting.

        This method is called by `BasePSF.view()` to determine axis labels.
        It computes the total spatial width and height (in micrometers) of the
        provided PSF image data.

        The calculation uses:
        - Optic's effective F-number (FNO_eff).
        - Wavelength of light (using `self.wavelengths[0]`, as scale is typically
          set by one reference wavelength, units are in µm).
        - Q-factor: Ratio of `self.grid_size` (FFT grid size) to
          `self.num_rays` (pupil sampling density).

        Args:
            image_data_for_shape (be.ndarray): The PSF image data (often a
                zoomed/cropped version from `BasePSF.view`). Its shape is used
                to determine the total extent for labeling.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the physical
            total width and total height of the PSF image area, in micrometers.
            These are returned as NumPy arrays as `BasePSF.view` expects them
            for Matplotlib's `extent` argument.
        """
        # Use paraxial F-number (FNO) as a starting point.
        fno_paraxial = self.optic.paraxial.FNO()

        # Adjust FNO for finite conjugates if the object is not at infinity.
        fno_effective = fno_paraxial
        if not self.optic.object_surface.is_infinite:
            exit_pupil_diameter = self.optic.paraxial.XPD()
            entrance_pupil_diameter = self.optic.paraxial.EPD()
            
            if entrance_pupil_diameter == 0: # Avoid division by zero
                 # Default to pupil magnification of 1 if EPD is zero (e.g. afocal)
                 # Or consider raising an error or specific handling.
                 pupil_magnification = 1.0
            else:
                pupil_magnification = exit_pupil_diameter / entrance_pupil_diameter
            
            magnification = self.optic.paraxial.magnification()
            # Effective F-number formula for finite conjugates
            fno_effective = fno_paraxial * (1 + be.abs(magnification) / pupil_magnification)

        # Q factor: ratio of FFT grid size to pupil sampling resolution.
        # A larger Q means finer sampling of the PSF (more pixels per Airy disk).
        if self.num_rays == 0: # Avoid division by zero if num_rays is invalid
            q_factor = 1.0 # Default or raise error
        else:
            q_factor = self.grid_size / self.num_rays # Both are int, ensure float division if needed by backend
            q_factor = be.astype(q_factor, be.float_dtype())
        
        # Pixel size (delta_x_psf) in the PSF image plane, in micrometers.
        # self.wavelengths[0] is in micrometers.
        if q_factor == 0: # Avoid division by zero
            pixel_size_um = be.array(0.0, dtype=be.float_dtype())
        else:
            pixel_size_um = self.wavelengths[0] * fno_effective / q_factor

        # Total extent in x (width) and y (height) in micrometers.
        # image_data_for_shape is the (potentially zoomed) PSF data passed from view().
        num_cols = image_data_for_shape.shape[1]
        num_rows = image_data_for_shape.shape[0]
        
        x_extent_total_um = num_cols * pixel_size_um
        y_extent_total_um = num_rows * pixel_size_um

        # BasePSF.view expects numpy arrays for matplotlib's extent argument.
        return be.to_numpy(x_extent_total_um), be.to_numpy(y_extent_total_um)

    # The strehl_ratio() method is inherited from BasePSF.
    # The BasePSF.strehl_ratio() default implementation is:
    #   center_idx = self.grid_size // 2
    #   return self.psf[center_idx, center_idx] / 100.0
    # This is suitable for FFTPSF because:
    # 1. self.psf is normalized by _compute_psf so a diffraction-limited
    #    peak is 100.0.
    # 2. The FFT process with fftshift centers the PSF, so the peak for an
    #    on-axis field point is at grid_size // 2.
    # No override is needed here.

# Removed methods that are now in BasePSF:
# - view
# - _plot_2d
# - _plot_3d
# - _log_tick_formatter
# - _log_colorbar_formatter
# - _interpolate_psf (BasePSF._interpolate_psf uses be.to_numpy and scipy.ndimage.zoom)
# - _find_bounds (BasePSF._find_bounds uses be.to_numpy and numpy operations)
