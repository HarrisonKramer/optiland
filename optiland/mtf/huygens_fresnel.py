"""Huygens-Fresnel based Modulation Transfer Function (MTF) Module

This module provides the HuygensMTF class for calculating the MTF based on
PSFs computed using the Huygens-Fresnel principle.
"""

import optiland.backend as be
from optiland.psf import HuygensPSF
from optiland.mtf.base import BaseMTF # Assuming BaseMTF is in base.py

class HuygensMTF(BaseMTF):
    """Huygens-Fresnel based Modulation Transfer Function (MTF).

    Calculates MTF by FFT of a PSF from `HuygensPSF`.
    `HuygensPSF` and therefore `HuygensMTF` currently only support the 'numpy'
    backend.

    Inherits common setup and plotting from `BaseMTF`.

    Args:
        optic (Optic): The optical system.
        fields (str or list): Field points for MTF calculation.
        wavelength (str or float): Wavelength for MTF calculation.
        num_rays (int, optional): Number of rays to sample pupil for HuygensPSF.
                              Defaults to BaseMTF's default (128).
        image_size (int, optional): Image size for PSF calculation in HuygensPSF.
                                 Defaults to 128 (as in HuygensPSF).
        num_points (int, optional): Number of points for MTF curve.
                                Defaults to BaseMTF's default (256).
        max_freq (str or float, optional): Maximum frequency.
                                       Defaults to BaseMTF's default ('cutoff').

    Attributes:
        image_size (int): Image size used for `HuygensPSF`.
        psf_cache (dict): Cache for computed PSFs.
    """

    def __init__(
        self,
        optic,
        fields,
        wavelength,
        num_rays=128,    # Consistent with BaseMTF default
        image_size=128,  # HuygensPSF default
        num_points=256,  # Consistent with BaseMTF default
        max_freq="cutoff",
    ):
        if be.get_backend() != "numpy":
            raise ValueError("HuygensMTF (and HuygensPSF) only supports the 'numpy' backend at this time.")

        super().__init__(
            optic=optic,
            fields=fields,
            wavelength=wavelength,
            max_freq=max_freq,
            num_points=num_points,
            num_rays=num_rays,
            # grid_size is not directly used by HuygensPSF, but BaseMTF stores it.
            # Pass a default or num_rays, it's not critical for this class.
            grid_size=image_size # Or some other logical default for BaseMTF's grid_size
        )

        self.image_size = image_size # Specific to HuygensPSF
        self.psf_cache = {}

        self.mtf_data = self._generate_mtf_data()

    def _get_psf_for_field(self, field_coords):
        """Computes or retrieves from cache the PSF using HuygensPSF.

        Args:
            field_coords (tuple): The (Hx, Hy) coordinates of the field.

        Returns:
            tuple[be.ndarray, float]: The computed PSF (from HuygensPSF) and
                                      its pixel_pitch. Returns (None, None)
                                      if PSF computation failed.
        """
        cache_key = (field_coords, self.wavelength, self.num_rays, self.image_size)
        if cache_key not in self.psf_cache:
            # HuygensPSF uses self.wavelength, self.num_rays, self.image_size
            psf_calculator = HuygensPSF(
                optic=self.optic,
                field=field_coords,
                wavelength=self.wavelength, # Single float from BaseMTF
                num_rays=self.num_rays,     # From BaseMTF
                image_size=self.image_size
            )
            # HuygensPSF.psf can be None if computation fails (e.g. no rays hit detector)
            if psf_calculator.psf is None:
                # Cache the failure to avoid re-computation
                self.psf_cache[cache_key] = None
                self.psf_cache[cache_key + ('pixel_pitch',)] = None
                return None, None

            self.psf_cache[cache_key] = psf_calculator.psf
            # Store pixel_pitch from HuygensPSF, needed for frequency scaling
            self.psf_cache[cache_key + ('pixel_pitch',)] = psf_calculator.pixel_pitch

        return self.psf_cache[cache_key], self.psf_cache[cache_key + ('pixel_pitch',)]


    def _calculate_otf_from_psf(self, psf_data):
        """Calculates OTF from PSF data (FFT of PSF)."""
        # PSF from HuygensPSF is real intensity.
        otf_complex = be.fft.fftshift(be.fft.fft2(psf_data))
        return otf_complex

    def _extract_mtf_profiles(self, otf_complex, psf_pixel_pitch_mm):
        """Extracts and interpolates Tangential and Sagittal MTF profiles.

        Args:
            otf_complex (be.ndarray): The 2D complex OTF, centered.
            psf_pixel_pitch_mm (float): The pixel pitch of the PSF in mm,
                                        from HuygensPSF.

        Returns:
            tuple[be.ndarray, be.ndarray]: (tangential_mtf, sagittal_mtf)
                                           interpolated onto self.freq.
        """
        mtf_2d = be.abs(otf_complex)
        center_y, center_x = mtf_2d.shape[0] // 2, mtf_2d.shape[1] // 2

        raw_tangential = mtf_2d[center_y:, center_x]
        raw_sagittal = mtf_2d[center_y, center_x:]

        dc_value = mtf_2d[center_y, center_x]
        if dc_value == 0:
            return be.zeros_like(self.freq), be.zeros_like(self.freq)

        norm_tangential = raw_tangential / dc_value
        norm_sagittal = raw_sagittal / dc_value

        # Frequency scaling for OTF from HuygensPSF:
        # HuygensPSF has image_size (N) points. PSF pixel pitch is psf_pixel_pitch_mm.
        # Total spatial width of PSF W_mm = N * psf_pixel_pitch_mm.
        # Frequency step in OTF is df_otf = 1 / W_mm.
        if psf_pixel_pitch_mm is None or psf_pixel_pitch_mm == 0 or self.image_size == 0 :
             f_step_otf = 0 # Avoid division by zero
        else:
            W_mm = self.image_size * psf_pixel_pitch_mm
            if W_mm == 0:
                f_step_otf = 0
            else:
                f_step_otf = 1.0 / W_mm
        
        # Raw frequency axis for the OTF data
        # Number of points in raw profile is self.image_size // 2
        num_raw_freq_points = self.image_size // 2
        raw_freq_axis = be.arange(num_raw_freq_points) * f_step_otf

        np_freq_target = be.to_numpy(self.freq)
        np_raw_freq_axis = be.to_numpy(raw_freq_axis)
        
        if len(np_raw_freq_axis) == 0 or f_step_otf == 0:
            interp_tangential = be.zeros_like(self.freq)
            interp_sagittal = be.zeros_like(self.freq)
        else:
            # Ensure raw profiles are correct length for raw_freq_axis
            # raw_sagittal (and tangential) length should be num_raw_freq_points
            np_norm_tangential = be.to_numpy(norm_tangential[:num_raw_freq_points])
            np_norm_sagittal = be.to_numpy(norm_sagittal[:num_raw_freq_points])

            interp_t_np = be.np.interp(np_freq_target, np_raw_freq_axis, np_norm_tangential, right=0.0)
            interp_s_np = be.np.interp(np_freq_target, np_raw_freq_axis, np_norm_sagittal, right=0.0)

            interp_tangential = be.array(interp_t_np)
            interp_sagittal = be.array(interp_s_np)
            
        return interp_tangential, interp_sagittal

    def _generate_mtf_data(self):
        """Generates MTF data using HuygensPSF and FFT.

        Implements abstract method from BaseMTF.
        """
        if be.get_backend() != "numpy":
            # Redundant check if __init__ has it, but good for safety if called directly.
            raise ValueError("HuygensMTF only supports the 'numpy' backend.")

        calculated_mtfs = []
        for field_coords in self.fields:
            psf, psf_pixel_pitch = self._get_psf_for_field(field_coords)
            if psf is None or psf_pixel_pitch is None:
                 # This means PSF calculation failed (e.g. no rays, or other HuygensPSF issue)
                 # Return zero MTF for this field.
                 calculated_mtfs.append([be.zeros_like(self.freq), be.zeros_like(self.freq)])
                 continue
            otf = self._calculate_otf_from_psf(psf)
            mtf_t, mtf_s = self._extract_mtf_profiles(otf, psf_pixel_pitch)
            calculated_mtfs.append([mtf_t, mtf_s])
        return calculated_mtfs

```
