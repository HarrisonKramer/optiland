# optiland/analysis/through_focus_mtf.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

import optiland.backend as be
from optiland.mtf import SampledMTF
from optiland.analysis.through_focus import ThroughFocusAnalysis

class ThroughFocusMTF(ThroughFocusAnalysis):
    """
    Performs Modulation Transfer Function (MTF) analysis across a range of focal positions.

    This class calculates the MTF at a specified spatial frequency for both
    tangential and sagittal orientations at multiple focal planes around the
    nominal focus of an optical system.

    The results include tangential and sagittal MTF values for each analyzed
    field at each focal step.
    """

    def __init__(
        self,
        optic,
        spatial_frequency, # type: float
        delta_focus=0.1,   # type: float
        num_steps=5,       # type: int
        fields="all",      # type: list[tuple[float, float]] | str
        wavelength="primary", # type: float | str
        num_rays=64,       # type: int
    ):
        """
        Initializes the ThroughFocusMTF analysis.

        Args:
            optic: The optiland.optic.Optic object to analyze.
            spatial_frequency (float): The single spatial frequency (in cycles/mm)
                at which to calculate MTF. The calculation will be performed
                for both tangential (fx = spatial_frequency, fy = 0) and
                sagittal (fx = 0, fy = spatial_frequency) orientations.
            delta_focus (float, optional): The increment of focal shift in mm.
                Defaults to 0.1.
            num_steps (int, optional): The number of focal planes to analyze
                before and after the nominal focus. Must be an odd integer.
                Defaults to 5.
            fields (list[tuple[float, float]] | str, optional): Fields for
                analysis. If "all", uses all fields from `optic.fields`.
                Defaults to "all".
            wavelength (float | str, optional): The wavelength (in µm) for
                analysis. If "primary", uses the primary wavelength from
                `optic.primary_wavelength`. Defaults to "primary".
            num_rays (int, optional): The number of rays across the pupil in 1D
                for the SampledMTF calculation. Defaults to 64.
        """
        self.spatial_frequency = spatial_frequency
        self.num_rays = num_rays

        if wavelength == "primary":
            # Assuming optic.primary_wavelength gives the value,
            # and optic.wavelengths.primary_wavelength might be another way if available
            # Based on SampledMTF, optic.primary_wavelength is the way
            resolved_self_wavelength = optic.primary_wavelength
        else:
            resolved_self_wavelength = wavelength

        self.wavelength = resolved_self_wavelength # Store the resolved wavelength

        super().__init__(
            optic,
            delta_focus=delta_focus,
            num_steps=num_steps,
            fields=fields,
            wavelengths=[self.wavelength], # Pass resolved wavelength as a single-element list
        )

    def _perform_analysis_at_focus(self):
        """
        Performs the MTF analysis at the current focal position for all fields.

        This method is called by the base class for each focal step. It
        calculates the tangential and sagittal MTF values for the specified
        spatial frequency for each field defined in `self.fields`.

        Returns:
            list[dict[str, float]]: A list of dictionaries, where each
            dictionary corresponds to a field and contains the tangential
            and sagittal MTF values, e.g.,
            [{'tangential': 0.5, 'sagittal': 0.45},  # Field 1
             {'tangential': 0.3, 'sagittal': 0.28}]  # Field 2
        """
        analysis_results_at_this_focus = []
        for field_coord in self.fields:
            sampled_mtf = SampledMTF(
                optic=self.optic,
                field=field_coord,
                wavelength=self.wavelength,
                num_rays=self.num_rays,
                distribution="uniform",
                zernike_terms=37,
                zernike_type="fringe",
            )

            freq_tan = (self.spatial_frequency, 0.0)
            freq_sag = (0.0, self.spatial_frequency)

            mtf_t = sampled_mtf.calculate_mtf([freq_tan])[0]
            mtf_s = sampled_mtf.calculate_mtf([freq_sag])[0]

            analysis_results_at_this_focus.append(
                {'tangential': mtf_t, 'sagittal': mtf_s}
            )
        return analysis_results_at_this_focus

    def _validate_view_prerequisites(self):
        """Checks if prerequisites for viewing results are met."""
        if not hasattr(self, 'results') or not self.results:
            print("No data to display. Run analysis first.")
            return False
        if not hasattr(self, 'fields') or not self.fields or len(self.fields) == 0:
            print("No fields to plot.")
            return False
        # Also check if num_steps is consistent, though results structure depends on it
        if hasattr(self, 'num_steps') and len(self.results) != self.num_steps:
            print(f"Data inconsistency: Expected {self.num_steps} result entries, found {len(self.results)}.")
            return False
        # Check if each result entry has data for all fields
        if hasattr(self, 'fields') and hasattr(self, 'results') and self.results:
            for i_pos_result in self.results:
                if len(i_pos_result) != len(self.fields):
                    print(f"Data inconsistency: Expected {len(self.fields)} field results per focus step, found {len(i_pos_result)}.")
                    return False
        return True

    def view(self):
        """
        Visualizes the through-focus MTF results.

        This method plots the tangential and sagittal MTF values against
        defocus for each analyzed field. Spline smoothing is applied to
        the MTF data for a smoother curve if enough data points are available.
        The plot shows MTF at the spatial frequency defined during initialization.
        """
        if not self._validate_view_prerequisites():
            return

        fig, ax = plt.subplots()

        # self.positions and self.nominal_focus are set by the base class `ThroughFocusAnalysis`
        # self.nominal_focus is the central focus position from self.positions
        np_positions = be.to_numpy(be.asarray(self.positions))
        np_nominal_focus = be.to_numpy(be.asarray(self.nominal_focus))
        defocus_values_np = np_positions - np_nominal_focus

        # Using default matplotlib color cycling.
        # If specific colors per field are needed:
        # num_total_fields = len(self.fields)
        # colors = plt.cm.viridis(np.linspace(0, 1, num_total_fields))

        for i_field, field_coord in enumerate(self.fields):
            # if specific colors: current_color = colors[i_field]

            # Extract MTF values for the current field across all defocus positions
            # self.results is a list (num_steps long) of lists (num_fields long) of dicts
            mtf_t_values = be.to_numpy(be.asarray([
                self.results[i_pos][i_field]['tangential'] for i_pos in range(self.num_steps)
            ]))
            mtf_s_values = be.to_numpy(be.asarray([
                self.results[i_pos][i_field]['sagittal'] for i_pos in range(self.num_steps)
            ]))

            num_data_points = len(defocus_values_np)

            # Determine spline order k based on number of points
            if num_data_points >= 4: # Need at least k+1 points for spline of degree k
                k = 3 # Cubic spline
            elif num_data_points >= 2:
                k = 1 # Linear spline
            else:
                k = 0 # No spline, just plot points

            if k > 0 and num_data_points > k : # Ensure num_data_points is strictly greater than k for make_interp_spline
                # np.unique sorts the x values, which is required for make_interp_spline
                # Also, handle cases where all x values are the same (e.g., num_steps = 1)
                unique_defocus, unique_indices = np.unique(defocus_values_np, return_index=True)
                if len(unique_defocus) < 2 : # Not enough unique points for spline
                    k = 0 # Fallback to plotting raw points
                else:
                    # Use only unique points for spline fitting to avoid issues
                    mtf_t_unique = mtf_t_values[unique_indices]
                    mtf_s_unique = mtf_s_values[unique_indices]

                    defocus_smooth = np.linspace(unique_defocus.min(), unique_defocus.max(), 300)

                    spl_t = make_interp_spline(unique_defocus, mtf_t_unique, k=k, check_finite=False)
                    mtf_t_smooth = spl_t(defocus_smooth)

                    spl_s = make_interp_spline(unique_defocus, mtf_s_unique, k=k, check_finite=False)
                    mtf_s_smooth = spl_s(defocus_smooth)

                    ax.plot(defocus_smooth, np.clip(mtf_t_smooth,0,1), linestyle='-', label=f"Field {field_coord} T")
                    ax.plot(defocus_smooth, np.clip(mtf_s_smooth,0,1), linestyle='--', label=f"Field {field_coord} S")
            else: # Not enough points for any spline or fallback, plot raw data
                k = 0 # Explicitly set k=0

            if k == 0: # Plot raw data if spline conditions not met or k was initially 0
                ax.plot(defocus_values_np, np.clip(mtf_t_values,0,1), linestyle='-', marker='o', markersize=4, label=f"Field {field_coord} T (raw)")
                ax.plot(defocus_values_np, np.clip(mtf_s_values,0,1), linestyle='--', marker='x', markersize=4, label=f"Field {field_coord} S (raw)")

        ax.set_title(f"Through-Focus MTF @ {self.spatial_frequency} cyc/mm, λ={self.wavelength:.3f} µm")
        ax.set_xlabel("Defocus (mm)")
        ax.set_ylabel("MTF")
        ax.set_ylim([0, 1.05])
        ax.legend(loc='best')
        ax.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()
        plt.show()
