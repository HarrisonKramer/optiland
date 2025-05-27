"""Base MTF Module

This module provides a base class for Modulation Transfer Function (MTF)
calculations.

Kramer Harrison, 2025 (Assumed author based on other files)
"""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import optiland.backend as be

class BaseMTF(ABC):
    """Base class for Modulation Transfer Function (MTF) calculations.

    This class provides common functionalities for MTF computation, including
    frequency axis generation, diffraction limit calculation, and plotting.
    Subclasses are expected to implement their specific MTF data generation logic.

    Args:
        optic (Optic): The optical system.
        fields (str or list): The field points at which to compute the MTF.
            If 'all', uses all fields defined in the optic.
        wavelength (str or float): The wavelength of light in micrometers.
            If 'primary', uses the optic's primary wavelength.
        max_freq (str or float, optional): The maximum spatial frequency for
            MTF calculation, in cycles/mm. If 'cutoff', it's calculated
            based on the diffraction limit (1 / (wavelength_mm * FNO)).
            Defaults to 'cutoff'.
        num_points (int, optional): The number of points to sample along the
            frequency axis for the MTF curve. Defaults to 256.
        num_rays (int, optional): The number of rays used for underlying PSF
            or spot diagram calculations if applicable. Defaults to 128.
            This might be used differently or overridden by subclasses.
        grid_size (int, optional): The size of the grid used for FFT-based
            PSF calculations if applicable. Defaults to 1024.
            This might be used differently or overridden by subclasses.


    Attributes:
        optic (Optic): The optical system.
        fields (list): List of field coordinate tuples (Hx, Hy).
        wavelength (float): Wavelength in micrometers.
        max_freq (float): Maximum frequency in cycles/mm.
        num_points (int): Number of frequency points.
        num_rays (int): Number of rays (for subclasses).
        grid_size (int): Grid size (for subclasses).
        FNO (float): Effective F-number of the optic.
        freq (be.ndarray): Array of spatial frequencies in cycles/mm.
        diff_limited_mtf (be.ndarray): Theoretical diffraction-limited MTF
            at the specified frequencies.
        mtf_data (list): List to be populated by subclasses, containing
            MTF results for each field. Each item should typically be a
            list or tuple [tangential_mtf, sagittal_mtf].
    """

    def __init__(self, optic, fields, wavelength, max_freq="cutoff", num_points=256, num_rays=128, grid_size=1024):
        self.optic = optic
        self._parse_fields(fields) # Sets self.fields
        self._parse_wavelength(wavelength) # Sets self.wavelength

        self.num_points = num_points
        self.num_rays = num_rays # May be used by subclasses
        self.grid_size = grid_size # May be used by subclasses

        self.FNO = self._get_effective_FNO()
        self._parse_max_freq(max_freq) # Sets self.max_freq

        self.freq = be.linspace(0, self.max_freq, self.num_points)
        self.diff_limited_mtf = self._calculate_diffraction_limit()

        self.mtf_data = None # To be computed by subclasses via _generate_mtf_data

    def _parse_fields(self, fields_arg):
        """Parses the fields argument."""
        if fields_arg == "all":
            self.fields = self.optic.fields.get_field_coords()
            if not self.fields: # Ensure there's at least one field
                 self.fields = [(0,0)]
        elif isinstance(fields_arg, list):
            self.fields = fields_arg
        elif isinstance(fields_arg, tuple) and len(fields_arg) == 2:
            self.fields = [fields_arg]
        else:
            raise ValueError("Invalid 'fields' argument. Must be 'all', a list of tuples, or a single tuple.")

    def _parse_wavelength(self, wavelength_arg):
        """Parses the wavelength argument."""
        if wavelength_arg == "primary":
            self.wavelength = self.optic.primary_wavelength
        elif isinstance(wavelength_arg, (float, int)):
            self.wavelength = float(wavelength_arg)
        else:
            raise ValueError("Invalid 'wavelength' argument. Must be 'primary' or a numeric value.")

    def _parse_max_freq(self, max_freq_arg):
        """Parses the max_freq argument."""
        if max_freq_arg == "cutoff":
            wavelength_mm = self.wavelength * 1e-3 # Convert µm to mm
            if self.FNO == 0: # Avoid division by zero for afocal systems or error
                # Default to a reasonable max frequency if FNO is zero, e.g. 100 cycles/mm
                # Or raise an error, as cutoff is ill-defined.
                # For now, let's use a default, but this might need review.
                self.max_freq = 100.0
                # Consider warning the user:
                # import warnings
                # warnings.warn("FNO is zero, 'cutoff' frequency is ill-defined. Defaulting max_freq to 100 cycles/mm.")
            else:
                self.max_freq = 1.0 / (wavelength_mm * self.FNO)
        elif isinstance(max_freq_arg, (float, int)):
            self.max_freq = float(max_freq_arg)
        else:
            raise ValueError("Invalid 'max_freq' argument. Must be 'cutoff' or a numeric value.")

    def _get_effective_FNO(self):
        """Calculates the effective F-number of the optical system.

        Returns:
            float: The effective F-number.
        """
        # This logic is common in PSF/MTF calculations.
        # Adapted from existing FFTPSF/_get_effective_FNO and FFTMTF/_get_fno
        fno_paraxial = self.optic.paraxial.FNO()

        if self.optic.object_surface.is_infinite:
            return fno_paraxial
        else:
            # Correction for finite conjugates
            # Using same logic as FFTPSF for consistency
            # For finite conjugates, FNO_eff = FNO_inf * (1 + |m|/p)
            # where p = XPD / EPD (pupil magnification for finite object)
            # This can also be expressed as FNO_eff = 1 / (2 * NA_image_space)
            # And NA_image_space = n_image * sin(theta_marginal_image)
            # Optic.paraxial.FNO() should ideally provide the working FNO.
            # Let's assume optic.paraxial.FNO() is the working FNO.
            # If not, the correction below is needed.
            # For now, assume optic.paraxial.FNO() IS the effective/working FNO.
            # If detailed correction is needed:
            # D_exit_pupil = self.optic.paraxial.XPD() # Exit pupil diameter
            # EPD = self.optic.paraxial.EPD() # Entrance pupil diameter
            # if EPD == 0: return fno_paraxial # Avoid division by zero
            # pupil_magnification_p = D_exit_pupil / EPD
            # magnification_m = self.optic.paraxial.magnification()
            # fno_effective = fno_paraxial * (1 + be.abs(magnification_m) / pupil_magnification_p)
            # return fno_effective
            # Re-evaluating: Optiland's paraxial.FNO() is typically the infinite conjugate FNO.
            # The correction for finite conjugates is usually FNO_working = FNO_infinite * (1 + |m|) for distant objects
            # or more generally related to NA.
            # Let's use the formula found in optiland.psf.base.BasePSF
            
            if not self.optic.object_surface.is_infinite:
                # This is the formula from BasePSF, assume it's correct for effective FNO
                D = self.optic.paraxial.XPD() # Exit Pupil Diameter
                # EPD = Entrance Pupil Diameter
                # p = D / EPD is pupil magnification
                # However, BasePSF uses self.optic.paraxial.EPD() for EPD
                # Let's assume paraxial.EPD gives the correct EPD for this formula
                if self.optic.paraxial.EPD() == 0: # Avoid division by zero
                    return fno_paraxial # or handle as error/warning

                p_val = D / self.optic.paraxial.EPD()
                if p_val == 0: # Avoid division by zero
                     return fno_paraxial

                m_val = self.optic.paraxial.magnification()
                return fno_paraxial * (1 + be.abs(m_val) / p_val)
            return fno_paraxial


    def _calculate_diffraction_limit(self):
        """Calculates the theoretical diffraction-limited MTF.

        Uses the formula: MTF(ν) = (2/π) * [arccos(ν/ν_c) - (ν/ν_c) * sqrt(1 - (ν/ν_c)^2)]
        where ν_c is the cutoff frequency.

        Returns:
            be.ndarray: The diffraction-limited MTF values corresponding to self.freq.
        """
        # Normalized frequency (freq / cutoff_freq)
        # cutoff_freq is self.max_freq if max_freq was set to 'cutoff'
        # Or, more generally, cutoff_freq = 1 / (wavelength_mm * FNO)
        wavelength_mm = self.wavelength * 1e-3
        if self.FNO == 0 : # Afocal system, or error
            # Diffraction limit is ill-defined or MTF is 1 everywhere up to some resolution limit
            # For now, return array of 1s, implies perfect resolution up to self.max_freq
            # This needs careful consideration for afocal systems.
            # import warnings
            # warnings.warn("FNO is zero, diffraction limit calculation may be inaccurate.")
            return be.ones_like(self.freq)

        cutoff_freq_theoretical = 1.0 / (wavelength_mm * self.FNO)
        if cutoff_freq_theoretical == 0: # Should not happen if FNO > 0 and wavelength > 0
            return be.ones_like(self.freq)

        norm_freq = self.freq / cutoff_freq_theoretical
        
        # Clip norm_freq to avoid issues with arccos for values slightly > 1 due to precision
        norm_freq = be.clip(norm_freq, 0, 1)

        # MTF formula
        term_acos = be.arccos(norm_freq)
        term_sqrt = be.sqrt(1 - norm_freq**2)
        mtf = (2 / be.pi) * (term_acos - norm_freq * term_sqrt)
        
        # Ensure MTF is 0 for frequencies >= cutoff_freq_theoretical
        # This is handled by norm_freq being clipped to 1, arccos(1)=0, sqrt(1-1^2)=0
        # So mtf becomes 0 at norm_freq = 1.
        
        # For frequencies beyond the theoretical cutoff, MTF should be zero.
        # The formula naturally handles this if norm_freq can exceed 1, then arccos becomes NaN.
        # Since we clipped norm_freq to 1, for self.freq > cutoff_freq_theoretical,
        # norm_freq is 1, and mtf is 0. This is correct.
        return mtf

    @abstractmethod
    def _generate_mtf_data(self):
        """Abstract method to compute MTF data.

        Subclasses must implement this method to calculate the MTF values
        (tangential and sagittal) for each field point and store them in
        self.mtf_data.

        The structure of self.mtf_data should be a list of items, where each
        item corresponds to a field in self.fields. Each item should typically
        be a list or tuple: [tangential_mtf_array, sagittal_mtf_array].
        """
        raise NotImplementedError("Subclasses must implement _generate_mtf_data.")

    def view(self, figsize=(10, 6), add_reference=True, title="Modulation Transfer Function"):
        """Plots the MTF curves for all fields.

        Args:
            figsize (tuple, optional): The figure size. Defaults to (10, 6).
            add_reference (bool, optional): Whether to plot the
                diffraction-limited MTF as a reference. Defaults to True.
            title (str, optional): The title for the plot. Defaults to
                "Modulation Transfer Function".
        """
        if self.mtf_data is None:
            # Compute MTF data if not already done (e.g., if user calls view() directly)
            self.mtf_data = self._generate_mtf_data()
            if self.mtf_data is None: # If subclass still didn't set it
                 raise RuntimeError("MTF data has not been computed by the subclass.")


        _, ax = plt.subplots(figsize=figsize)

        num_fields = len(self.fields)
        colors = plt.cm.get_cmap('viridis', num_fields) # Get a colormap

        for i, field_data in enumerate(self.mtf_data):
            if not (isinstance(field_data, (list, tuple)) and len(field_data) == 2):
                print(f"Warning: MTF data for field {self.fields[i]} is not in the expected format [tangential, sagittal]. Skipping.")
                continue

            tangential_mtf, sagittal_mtf = field_data
            field_label = f"Hx: {self.fields[i][0]:.2f}, Hy: {self.fields[i][1]:.2f}"
            color = colors(i / max(1, num_fields -1)) if num_fields > 1 else colors(0)


            ax.plot(
                be.to_numpy(self.freq),
                be.to_numpy(tangential_mtf),
                label=f"{field_label} - Tangential",
                color=color,
                linestyle="-"
            )
            ax.plot(
                be.to_numpy(self.freq),
                be.to_numpy(sagittal_mtf),
                label=f"{field_label} - Sagittal",
                color=color,
                linestyle="--"
            )

        if add_reference:
            ax.plot(
                be.to_numpy(self.freq),
                be.to_numpy(self.diff_limited_mtf),
                "k:", # Black dotted line for reference
                label="Diffraction Limit"
            )

        ax.set_xlabel("Spatial Frequency (cycles/mm)")
        ax.set_ylabel("Modulation")
        ax.set_title(title)
        ax.set_xlim([0, be.to_numpy(self.max_freq)])
        ax.set_ylim([0, 1.05]) # MTF typically up to 1
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # Place legend outside the plot
        ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        plt.show()

```
