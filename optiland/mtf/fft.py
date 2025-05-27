"""FFT-based Modulation Transfer Function (MTF) Module

This module provides the FFTMTF class for calculating the MTF based on
the Fast Fourier Transform of the Point Spread Function (PSF).
"""

import optiland.backend as be
from optiland.psf import FFTPSF
from optiland.mtf.base import BaseMTF


class FFTMTF(BaseMTF):
    """FFT-based Modulation Transfer Function (MTF).

    This class calculates the MTF of an optical system by performing a Fast
    Fourier Transform (FFT) on the Point Spread Function (PSF), where the PSF
    itself is computed using an FFT of the pupil function via `FFTPSF`.

    It inherits common setup (optic, fields, wavelength, frequency axis) and
    plotting functionalities from `BaseMTF`.

    Args:
        optic (Optic): The optical system.
        fields (str or list): Field points for MTF calculation.
        wavelength (str or float): Wavelength for MTF calculation.
        num_rays (int, optional): Number of rays to sample pupil for PSF.
                              Defaults to BaseMTF's default (128).
        grid_size (int, optional): Grid size for FFT in PSF calculation.
                               Defaults to BaseMTF's default (1024).
        num_points (int, optional): Number of points for MTF curve.
                                Defaults to BaseMTF's default (256).
        max_freq (str or float, optional): Maximum frequency.
                                       Defaults to BaseMTF's default ('cutoff').

    Attributes:
        psf_cache (dict): A cache to store computed PSFs for fields/wavelengths
                          to avoid recomputation if needed, though current structure
                          computes them once in _generate_mtf_data.
    """

    def __init__(
        self,
        optic,
        fields,
        wavelength,
        num_rays=128,   # Consistent with BaseMTF default
        grid_size=1024, # Consistent with BaseMTF default
        num_points=256, # Consistent with BaseMTF default
        max_freq="cutoff",
    ):
        super().__init__(
            optic=optic,
            fields=fields,
            wavelength=wavelength,
            max_freq=max_freq,
            num_points=num_points,
            num_rays=num_rays,
            grid_size=grid_size
        )
        # self.num_rays and self.grid_size are stored by BaseMTF and used here.
        # self.wavelength is the single float value from BaseMTF.
        # self.fields is the list of field tuples from BaseMTF.

        self.psf_cache = {} # Initialize PSF cache

        # Compute MTF data upon initialization
        self.mtf_data = self._generate_mtf_data()

    def _get_psf_for_field(self, field_coords):
        """Computes or retrieves from cache the PSF for a given field.

        Args:
            field_coords (tuple): The (Hx, Hy) coordinates of the field.

        Returns:
            be.ndarray: The computed PSF (from FFTPSF).
        """
        # Using wavelength from self.wavelength (parsed by BaseMTF)
        # Using num_rays and grid_size from self (parsed by BaseMTF)
        cache_key = (field_coords, self.wavelength, self.num_rays, self.grid_size)
        if cache_key not in self.psf_cache:
            psf_calculator = FFTPSF(
                optic=self.optic,
                field=field_coords,
                wavelength=self.wavelength,
                num_rays=self.num_rays,
                grid_size=self.grid_size,
            )
            self.psf_cache[cache_key] = psf_calculator.psf
        return self.psf_cache[cache_key]

    def _calculate_otf_from_psf(self, psf_data):
        """Calculates the Optical Transfer Function (OTF) from PSF data.

        The OTF is the Fourier Transform of the PSF.

        Args:
            psf_data (be.ndarray): The 2D Point Spread Function.

        Returns:
            be.ndarray: The 2D Optical Transfer Function, shifted so DC is at center.
        """
        # The PSF from FFTPSF is already real and represents intensity.
        # The OTF is the FFT of the PSF.
        otf_complex = be.fft.fftshift(be.fft.fft2(psf_data))
        return otf_complex
        
    def _extract_mtf_profiles(self, otf_complex):
        """Extracts Tangential and Sagittal MTF profiles from the OTF.

        The MTF is the magnitude of the OTF. Profiles are taken along axes
        through the center (DC component) of the OTF.

        Args:
            otf_complex (be.ndarray): The 2D complex OTF, centered.

        Returns:
            tuple[be.ndarray, be.ndarray]: (tangential_mtf, sagittal_mtf)
        """
        # MTF is the absolute value of the OTF
        mtf_2d = be.abs(otf_complex)

        # Determine center of the 2D MTF array
        center_x = mtf_2d.shape[1] // 2
        center_y = mtf_2d.shape[0] // 2
        
        raw_tangential = mtf_2d[center_y:, center_x]
        raw_sagittal = mtf_2d[center_y, center_x:]

        dc_value = mtf_2d[center_y, center_x]
        if dc_value == 0: 
            return be.zeros_like(self.freq), be.zeros_like(self.freq) # Return MTF of 0 for target freqs

        norm_tangential = raw_tangential / dc_value
        norm_sagittal = raw_sagittal / dc_value
        
        wavelength_mm = self.wavelength * 1e-3 # Convert self.wavelength (µm) to mm
        
        # Calculate frequency step for the raw OTF data
        # df_otf = 1 / (num_rays_pupil * wavelength_mm * FNO)
        # num_rays_pupil is self.num_rays (stored by BaseMTF, used by FFTPSF)
        # FNO is self.FNO (calculated by BaseMTF)
        if wavelength_mm == 0 or self.FNO == 0 or self.num_rays == 0:
            f_step_otf = 0 
        else:
            f_step_otf = 1.0 / (self.num_rays * wavelength_mm * self.FNO)
            
        # If grid_size is used in FFTPSF, it affects PSF pixel size, thus OTF frequency step.
        # The derivation: df_otf = 1 / (num_rays_pupil * wavelength_mm * FNO) is correct.
        # num_rays_pupil determines the spatial extent of the PSF that is sampled by grid_size pixels.
        # FFTPSF's Q factor: Q = grid_size_fft / num_rays_pupil_sampling
        # PSF pixel size: dx_psf = wavelength * FNO / Q
        # Total PSF width for FFT: W_psf = grid_size_fft * dx_psf = grid_size_fft * (wavelength * FNO / (grid_size_fft / num_rays_pupil))
        # W_psf = num_rays_pupil * wavelength * FNO.
        # OTF frequency step: df_otf = 1 / W_psf. This matches.

        # Generate the frequency axis for the raw MTF profiles
        # Assuming raw_tangential and raw_sagittal have same length originating from square PSF/OTF
        num_raw_points = len(raw_sagittal) # Number of points from DC to Nyquist in raw OTF
        raw_freq_axis = be.arange(num_raw_points) * f_step_otf

        np_freq_target = be.to_numpy(self.freq) # Target frequencies from BaseMTF
        np_raw_freq_axis = be.to_numpy(raw_freq_axis)
        
        if len(np_raw_freq_axis) == 0 or f_step_otf == 0:
            # Handle cases where raw data is empty or frequency step is zero
            interp_tangential = be.zeros_like(self.freq)
            interp_sagittal = be.zeros_like(self.freq)
        else:
            np_norm_tangential = be.to_numpy(norm_tangential)
            np_norm_sagittal = be.to_numpy(norm_sagittal)

            # Interpolate onto the target frequency axis self.freq
            # Points in np_freq_target outside range of np_raw_freq_axis will get value of right=0.0
            interp_t_np = be.np.interp(np_freq_target, np_raw_freq_axis, np_norm_tangential, right=0.0)
            interp_s_np = be.np.interp(np_freq_target, np_raw_freq_axis, np_norm_sagittal, right=0.0)

            interp_tangential = be.array(interp_t_np)
            interp_sagittal = be.array(interp_s_np)

        return interp_tangential, interp_sagittal


    def _generate_mtf_data(self):
        """Generates the MTF data for each field point using FFT of PSF.

        Implements the abstract method from BaseMTF.
        For each field:
        1. Computes the PSF using `FFTPSF`.
        2. Computes the OTF by FFTing the PSF.
        3. Extracts and normalizes tangential and sagittal MTF profiles from OTF.
        4. Interpolates these profiles onto the common frequency axis `self.freq`.

        Returns:
            list: A list of [tangential_mtf, sagittal_mtf] pairs for each field.
        """
        calculated_mtfs = []
        for field_coords in self.fields: # self.fields is list of (Hx,Hy) from BaseMTF
            psf = self._get_psf_for_field(field_coords)
            
            if psf is None or be.sum(psf) == 0: # Handle empty or zero PSF
                # If PSF is invalid, MTF is zero
                mtf_t = be.zeros_like(self.freq)
                mtf_s = be.zeros_like(self.freq)
            else:
                otf = self._calculate_otf_from_psf(psf)
                mtf_t, mtf_s = self._extract_mtf_profiles(otf)
            
            calculated_mtfs.append([mtf_t, mtf_s])
        return calculated_mtfs

```
