"""Radiant Intensity Analysis

This module implements the logic for radiant intensity analysis
in an optical system, representing power per unit solid angle.


Manuel Fragata Mendes, June 2025
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import optiland.backend as be
from .base import BaseAnalysis


class RadiantIntensity(BaseAnalysis):
    """
    Computes and visualizes the radiant intensity distribution.

    By default, this analysis calculates radiant intensity in absolute physical
    units of Watts/steradian (W/sr). This requires that the .intensity
    attribute of the rays traced represents the physical power of each ray.

    The angles correspond to projections, similar to Zemax's
    "Angle Space" plots (Angle X, Angle Y).

    Attributes:
        optic (optiland.optic.Optic): The optical system.
        num_angular_bins_X (int): Number of bins for the X-angle.
        num_angular_bins_Y (int): Number of bins for the Y-angle.
        angle_X_min (float): Minimum X-angle in degrees for binning.
        angle_X_max (float): Maximum X-angle in degrees for binning.
        angle_Y_min (float): Minimum Y-angle in degrees for binning.
        angle_Y_max (float): Maximum Y-angle in degrees for binning.
        reference_surface_index (int): Index of the surface *after* which ray
                                       directions are considered.
        fields (list): List of field coordinates for analysis.
        wavelengths (list): List of wavelengths for analysis.
        num_rays (int): Number of rays to trace if user_initial_rays is None.
        distribution_name (str): Ray distribution if user_initial_rays is None.
        user_initial_rays (RealRays | None): Optional user-provided initial rays.
        data (list[list[tuple]]): Stores (intensity_map,
                                          angle_X_bin_edges, angle_Y_bin_edges,
                                          angle_X_bin_centers, angle_Y_bin_centers)
                                  for each (field, wavelength).
        use_absolute_units (bool): If True (default), calculates intensity in W/sr.
                                   If False, result is a relative value normalized
                                   to the peak.
    """

    def __init__(
        self,
        optic,
        num_angular_bins_X: int = 101, 
        num_angular_bins_Y: int = 101, 
        angle_X_min: float = -15.0,
        angle_X_max: float = 15.0,
        angle_Y_min: float = -15.0,
        angle_Y_max: float = 15.0,
        use_absolute_units: bool = True, 
        reference_surface_index: int = -1,
        fields="all",
        wavelengths="all",
        num_rays: int = 100000,
        distribution: str = "random", # Default to importance sampling
        user_initial_rays=None,
    ):
        if fields == "all":
            self.fields = optic.fields.get_field_coords()
        else:
            if not isinstance(fields, list): fields = [fields]
            self.fields = tuple(fields)

        self.num_angular_bins_X = num_angular_bins_X
        self.num_angular_bins_Y = num_angular_bins_Y
        self.angle_X_min, self.angle_X_max = float(angle_X_min), float(angle_X_max)
        self.angle_Y_min, self.angle_Y_max = float(angle_Y_min), float(angle_Y_max)

        # for absolute units, we need to ensure the user has provided rays with 'calibrated' power
        self.use_absolute_units = use_absolute_units
        if self.use_absolute_units and user_initial_rays is None:
            print("Warning: `use_absolute_units` is True, but no `user_initial_rays` were provided.")
            print("         Internal ray generator may not have 'calibrated' power values.")
            print("         Resulting intensity map may not be in true W/sr.")
        

        self.reference_surface_index = int(reference_surface_index)
        self.num_rays = num_rays
        self.distribution_name = distribution
        self.user_initial_rays = user_initial_rays

        super().__init__(optic, wavelengths)
        
    def _generate_data(self):
        
        analysis_data = []
        for field_coord in self.fields:
            field_block = []
            for wl in self.wavelengths:
                field_block.append(
                    self._generate_field_wavelength_data(field_coord, wl)
                )
            analysis_data.append(field_block)
        return analysis_data

    def _generate_field_wavelength_data(self, field_coord, wavelength):
        
        if self.user_initial_rays is None:
            
            self.optic.trace(
                *field_coord, wavelength=wavelength,
                num_rays=self.num_rays, distribution=self.distribution_name
            )
        else:
            self.optic.surface_group.trace(self.user_initial_rays)

        
        surf_group = self.optic.surface_group
        try:
            ref_surf = surf_group.surfaces[self.reference_surface_index]
            L_all, M_all, N_all = ref_surf.L, ref_surf.M, ref_surf.N
            power_all = ref_surf.intensity
            if not (be.size(L_all) > 0): raise AttributeError
        except (IndexError, AttributeError):
            L_all, M_all, N_all, power_all = (be.empty(0) for _ in range(4))

        valid_mask = ((power_all > 1e-12) & ~be.isnan(L_all) & ~be.isnan(M_all) &
                      ~be.isnan(N_all) & (be.abs(N_all) > 1e-9))

        if not be.any(valid_mask):
            L_f, M_f, N_f, power_f = (be.empty(0) for _ in range(4))
        else:
            L_f, M_f, N_f, power_f = (arr[valid_mask] for arr in [L_all, M_all, N_all, power_all])

        angle_X_deg = be.degrees(be.arctan2(L_f, N_f))
        angle_Y_deg = be.degrees(be.arctan2(M_f, N_f))

        angle_X_bins = be.linspace(self.angle_X_min, self.angle_X_max, self.num_angular_bins_X + 1)
        angle_Y_bins = be.linspace(self.angle_Y_min, self.angle_Y_max, self.num_angular_bins_Y + 1)
        angle_X_centers = (angle_X_bins[:-1] + angle_X_bins[1:]) / 2
        angle_Y_centers = (angle_Y_bins[:-1] + angle_Y_bins[1:]) / 2

        power_map, _, _ = be.histogram2d(
            angle_X_deg, angle_Y_deg,
            bins=[angle_X_bins, angle_Y_bins],
            weights=power_f
        )

        if self.use_absolute_units:
            delta_angle_X_rad = be.radians(angle_X_bins[1] - angle_X_bins[0])
            delta_angle_Y_rad = be.radians(angle_Y_bins[1] - angle_Y_bins[0])
            solid_angle_per_bin_sr = delta_angle_X_rad * delta_angle_Y_rad

            final_intensity_map = be.where(solid_angle_per_bin_sr > 1e-12,
                                           power_map / solid_angle_per_bin_sr,
                                           be.zeros_like(power_map))
        else:
            final_intensity_map = power_map

        return (final_intensity_map, angle_X_bins, angle_Y_bins,
                angle_X_centers, angle_Y_centers)


    def peak_intensity_values(self):
        
        peaks = []
        if not self.data: return peaks
        for field_block in self.data:
            field_peaks = [be.max(entry[0]) if be.to_numpy(entry[0]).size > 0 else 0.0 for entry in field_block]
            peaks.append(field_peaks)
        return peaks

    def view(self, figsize=(14, 6), cmap="jet", cross_section_style='-', cross_section_color='red'):
        """
        Displays radiant intensity maps and a central cross-section.
        The y-axis and colorbar show absolute units (W/sr) if
        `use_absolute_units` was True, otherwise they show normalized values.
        """
        if not self.data:
            print("No intensity data to display.")
            return

        num_fields, num_wavelengths = len(self.fields), len(self.wavelengths)
        if num_fields == 0 or num_wavelengths == 0: return

        all_peak_values = self.peak_intensity_values()
        global_max_val = 0.0
        if self.use_absolute_units: # find global max for absolute scale
             if all_peak_values:
                global_max_val = max(max(be.to_numpy(p) for p in field_peaks) for field_peaks in all_peak_values)
             if global_max_val == 0: global_max_val = 1.0

        for f_idx in range(num_fields):
            for w_idx in range(num_wavelengths):
                fig = plt.figure(figsize=figsize)
                gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1.5], figure=fig)
                ax_map, ax_cs = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
                
                intensity_map_be, x_bins_be, y_bins_be, x_centers_be, y_centers_be = self.data[f_idx][w_idx]
                
                intensity_map = be.to_numpy(intensity_map_be)
                x_bins, y_bins = be.to_numpy(x_bins_be), be.to_numpy(y_bins_be)
                x_centers, y_centers = be.to_numpy(x_centers_be), be.to_numpy(y_centers_be)

                current_display_map = intensity_map
                
                if self.use_absolute_units:
                    vmin_plot, vmax_plot = 0.0, global_max_val
                    cbar_label = "Radiant Intensity (W/sr)"
                else: # normalize to peak for relative plot
                    peak_val = be.to_numpy(all_peak_values[f_idx][w_idx])
                    if peak_val > 1e-9:
                        current_display_map = intensity_map / peak_val
                    vmin_plot, vmax_plot = 0.0, 1.0
                    cbar_label = "Normalized Intensity"

                # left subplot: imshow
                im = ax_map.imshow(current_display_map.T, aspect="auto", origin="lower",
                                   extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                                   cmap=cmap, vmin=vmin_plot, vmax=vmax_plot)
                ax_map.set_xlabel("X-Angle (degrees)"); ax_map.set_ylabel("Y-Angle (degrees)")
                fig.colorbar(im, ax=ax_map, label=cbar_label, fraction=0.046, pad=0.04)

                # ight subplot: cross-section
                central_row_index = current_display_map.shape[1] // 2
                cross_section_data = current_display_map[:, central_row_index]
                ax_cs.plot(x_centers, cross_section_data, linestyle=cross_section_style, color=cross_section_color)
                ax_cs.set_xlabel("X-Angle (degrees)"); ax_cs.set_ylabel(cbar_label)
                ax_cs.grid(True, linestyle=':', alpha=0.7)
                ax_cs.set_xlim(x_centers[0], x_centers[-1])
                ax_cs.set_ylim(bottom=-0.05 * vmax_plot, top=vmax_plot * 1.1)
                
                title_str = f"Field: {self.fields[f_idx]}, Wavelength: {self.wavelengths[w_idx]:.3f} µm"
                fig.suptitle(f"Radiant Intensity\n{title_str}", fontsize=14)
                plt.tight_layout(rect=[0, 0.03, 1, 0.94])
                plt.show()
