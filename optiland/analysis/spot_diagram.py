"""Spot Diagram Analysis

This module provides a spot diagram analysis for optical systems.

Kramer Harrison, 2024
"""

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np # Ensure numpy is imported if used directly (e.g. np.abs)

import optiland.backend as be # [cite: uploaded:spot_diagram.py]
from optiland.visualization.utils import transform # [cite: uploaded:spot_diagram.py]


@dataclass
class SpotData: # [cite: uploaded:spot_diagram.py]
    """Stores the x, y coordinates and intensity of a spot.

    Attributes:
        x: Array of x-coordinates.
        y: Array of y-coordinates.
        intensity: Array of intensity values.
    """

    x: be.array # [cite: uploaded:spot_diagram.py]
    y: be.array # [cite: uploaded:spot_diagram.py]
    intensity: be.array # [cite: uploaded:spot_diagram.py]


class SpotDiagram: # [cite: uploaded:spot_diagram.py]
    """Spot diagram class

    This class generates data and plots real ray intersection locations
    on the final optical surface in an optical system. These plots
    are purely geometric and give an indication of the blur produced
    by aberrations in the system.

    Attributes:
        optic (optic.Optic): instance of the optic object to be assessed
        fields (tuple): fields at which data is generated
        wavelengths (tuple[float]): wavelengths at which data is generated
        num_rings (int): number of rings in pupil distribution for ray tracing
        data (List[List[SpotData]]): contains spot data in a nested list.
            Data is ordered as field (dim 0), wavelength (dim 1), then
            SpotData objects.
        coordinates (Literal['global', 'local']): Coordinate system for data
                                                  and plotting.

    """

    def __init__( # [cite: uploaded:spot_diagram.py]
        self,
        optic,
        fields="all",
        wavelengths="all",
        num_rings=6, # Default in Optiland's SpotDiagram was num_rays, but it's used as num_rings for trace
        distribution="hexapolar",
        coordinates: Literal["global", "local"] = "local",
    ):
        self.optic = optic # [cite: uploaded:spot_diagram.py]
        self.fields = fields # [cite: uploaded:spot_diagram.py]

        if coordinates not in ["global", "local"]: # [cite: uploaded:spot_diagram.py]
            raise ValueError("Coordinates must be 'global' or 'local'.") # [cite: uploaded:spot_diagram.py]
        self.coordinates = coordinates # [cite: uploaded:spot_diagram.py]

        self.wavelengths = wavelengths # [cite: uploaded:spot_diagram.py]
        if self.fields == "all": # [cite: uploaded:spot_diagram.py]
            self.fields = self.optic.fields.get_field_coords() # [cite: uploaded:spot_diagram.py]

        if self.wavelengths == "all": # [cite: uploaded:spot_diagram.py]
            self.wavelengths = self.optic.wavelengths.get_wavelengths() # [cite: uploaded:spot_diagram.py]
        
        # Ensure self.wavelengths is a list of float values if it was 'all' or 'primary'
        if isinstance(self.wavelengths, str): # Should have been resolved to list by get_wavelengths
             print(f"Warning (SpotDiagram init): Wavelengths is still '{self.wavelengths}', attempting to use primary.")
             self.wavelengths = [self.optic.wavelengths.primary_wavelength.value] if self.optic.wavelengths.primary_wavelength else [0.550]


        self.data = self._generate_data( # [cite: uploaded:spot_diagram.py]
            self.fields,
            self.wavelengths,
            num_rings, # Optiland's trace uses num_rays, SpotDiagram used num_rings for this
            distribution,
            self.coordinates,
        )

    def view(self, fig_to_plot_on=None, figsize=(12, 4), add_airy_disk=False): # [cite: uploaded:spot_diagram.py]
        """View the spot diagram

        Args:
            fig_to_plot_on (matplotlib.figure.Figure, optional): The Matplotlib
                figure object to plot on. If None, a new figure is created.
            figsize (tuple): the figure size of the output window.
                Default is (12, 4).
            add_airy_disk (bool): Airy disc visualization controller.

        Returns:
            None

        """
        is_gui_embedding = fig_to_plot_on is not None
        
        N = len(self.fields) # [cite: uploaded:spot_diagram.py]
        if N == 0:
            print("Warning (SpotDiagram.view): No fields to plot.")
            if is_gui_embedding and hasattr(fig_to_plot_on, 'canvas') and fig_to_plot_on.canvas is not None:
                fig_to_plot_on.text(0.5,0.5, "No fields to plot Spot Diagram", ha='center', va='center')
                fig_to_plot_on.canvas.draw_idle()
            return

        num_cols = 3 # As per original SpotDiagram
        num_rows = (N + num_cols - 1) // num_cols # Calculate rows needed

        if not is_gui_embedding:
            current_fig = plt.figure(figsize=(figsize[0], num_rows * figsize[1])) # [cite: uploaded:spot_diagram.py]
        else:
            current_fig = fig_to_plot_on
            current_fig.clear() # Clear the figure for new subplots

        # Create subplots on the current_fig
        # Note: fig.subplots() returns a Figure and an array of Axes.
        # If num_rows or num_cols is 1, it might return a single Axes or a 1D array.
        # We use sharex=True, sharey=True as in the original.
        try:
            # fig, axs = current_fig.subplots(num_rows, num_cols, sharex=True, sharey=True)
            # Using add_subplot is safer if num_rows=1 for axs.flatten() later
            axs_list = []
            for i in range(num_rows * num_cols):
                ax = current_fig.add_subplot(num_rows, num_cols, i + 1, sharex=axs_list[0] if i>0 and num_cols==1 else None, sharey=axs_list[0] if i>0 and num_cols==1 else None)
                if i > 0 and num_cols > 1 : # Manual sharing for grid
                    if (i % num_cols) != 0: # Share Y with left neighbor
                        ax.sharey(axs_list[-1])
                    if i >= num_cols: # Share X with top neighbor
                        ax.sharex(axs_list[i-num_cols])
                axs_list.append(ax)
            axs = np.array(axs_list).flatten()

        except Exception as e:
            print(f"Error creating subplots for Spot Diagram: {e}")
            if is_gui_embedding and hasattr(current_fig, 'canvas') and current_fig.canvas is not None:
                current_fig.text(0.5,0.5, f"Error creating subplots:\n{e}", ha='center', va='center', color='red')
                current_fig.canvas.draw_idle()
            return


        # Subtract centroid and find limits
        data = self._center_spots(self.data) # [cite: uploaded:spot_diagram.py]
        geometric_size = self.geometric_spot_radius() # [cite: uploaded:spot_diagram.py]
        
        axis_lim = 0.01 # Default small limit
        if geometric_size and any(any(s is not None for s in row) for row in geometric_size):
            try:
                # Filter out None before stacking/max
                valid_rows = []
                for row in geometric_size:
                    valid_elements = [s for s in row if s is not None]
                    if valid_elements:
                         valid_rows.append(be.stack(valid_elements, axis=0))
                
                if valid_rows:
                    gs_array = be.stack(valid_rows, axis=0) # [cite: uploaded:spot_diagram.py]
                    axis_lim = be.max(gs_array) # [cite: uploaded:spot_diagram.py]
                else:
                    print("Warning (SpotDiagram.view): All geometric sizes are None.")
            except Exception as e_gs:
                print(f"Warning (SpotDiagram.view): Could not determine axis_lim from geometric_size: {e_gs}")


        if add_airy_disk: # [cite: uploaded:spot_diagram.py]
            # Ensure primary wavelength is a float value
            primary_wl_obj = self.optic.wavelengths.primary_wavelength
            wavelength_val_for_airy = primary_wl_obj.value if primary_wl_obj else (self.wavelengths[0] if self.wavelengths else 0.550)

            centroids = self.centroid() # [cite: uploaded:spot_diagram.py]
            chief_ray_centers = self.generate_chief_rays_centers(wavelength=wavelength_val_for_airy) # [cite: uploaded:spot_diagram.py]
            airy_rad_x, airy_rad_y = self.airy_disc_x_y(wavelength=wavelength_val_for_airy) # [cite: uploaded:spot_diagram.py]
        else: # [cite: uploaded:spot_diagram.py]
            wavelength_val_for_airy = None # [cite: uploaded:spot_diagram.py]
            centroids = None # [cite: uploaded:spot_diagram.py]
            chief_ray_centers = None # [cite: uploaded:spot_diagram.py]
            airy_rad_x, airy_rad_y = None, None # [cite: uploaded:spot_diagram.py]

        # Plot wavelengths for each field
        for k, field_data in enumerate(data): # [cite: uploaded:spot_diagram.py]
            if k >= len(axs): break # Should not happen if num_rows, num_cols are correct
            
            real_centroid_x, real_centroid_y = None, None # [cite: uploaded:spot_diagram.py]
            if add_airy_disk and centroids and chief_ray_centers is not None and k < len(centroids) and k < len(chief_ray_centers): # [cite: uploaded:spot_diagram.py]
                real_centroid_x = chief_ray_centers[k][0] - centroids[k][0] # [cite: uploaded:spot_diagram.py]
                real_centroid_y = chief_ray_centers[k][1] - centroids[k][1] # [cite: uploaded:spot_diagram.py]
            
            self._plot_field( # [cite: uploaded:spot_diagram.py]
                axs[k],
                field_data,
                self.fields[k],
                axis_lim,
                self.wavelengths,
                add_airy_disk=add_airy_disk,
                field_index=k,
                airy_rad_x=airy_rad_x,
                airy_rad_y=airy_rad_y,
                real_centroid_x=real_centroid_x,
                real_centroid_y=real_centroid_y,
            )

        # Remove empty axes
        for k_ax in range(N, len(axs)): # [cite: uploaded:spot_diagram.py]
            current_fig.delaxes(axs[k_ax]) # [cite: uploaded:spot_diagram.py]

        # Attempt to create a single legend for the whole figure if multiple axes
        if N > 0 and len(axs) > 0:
            handles, labels = axs[0].get_legend_handles_labels()
            if handles: # If the first subplot has legend items
                 # Place legend outside, to the right of the subplots
                 # Adjust bbox_to_anchor and loc as needed. This is tricky with dynamic rows.
                 # For a fixed 3-column layout, this might work if placed carefully.
                 current_fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(self.wavelengths))


        current_fig.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust rect to make space for figure-level legend if used

        if not is_gui_embedding:
            plt.show() # [cite: uploaded:spot_diagram.py]
        else:
            if hasattr(current_fig, 'canvas') and current_fig.canvas is not None:
                current_fig.canvas.draw_idle()

    # ... (angle_from_cosine, f_number, airy_radius, generate_marginal_rays, etc. methods remain the same as in uploaded:spot_diagram.py)
    def angle_from_cosine(self, a, b): # [cite: uploaded:spot_diagram.py]
        """Calculate the angle (in radians) between two vectors given their
        direction cosine values.

        Returns:
            theta_rad (float): angle in radians.

        """
        a = a / be.linalg.norm(a) # [cite: uploaded:spot_diagram.py]
        b = b / be.linalg.norm(b) # [cite: uploaded:spot_diagram.py]
        theta = be.arccos(be.clip(be.dot(a, b), -1, 1)) # [cite: uploaded:spot_diagram.py]

        return theta # [cite: uploaded:spot_diagram.py]

    def f_number(self, n, theta): # [cite: uploaded:spot_diagram.py]
        """Calculates the F#

        Returns:
            N_w (float): Physical F number.

        """
        N_w = 1 / (2 * n * be.sin(theta)) # [cite: uploaded:spot_diagram.py]
        return N_w # [cite: uploaded:spot_diagram.py]

    def airy_radius(self, n_w, wavelength): # [cite: uploaded:spot_diagram.py]
        """Calculates the airy radius
        Returns:
            r (float): Airy radius.
        """
        r = 1.22 * n_w * wavelength # [cite: uploaded:spot_diagram.py]
        return r # [cite: uploaded:spot_diagram.py]

    def generate_marginal_rays(self, H_x, H_y, wavelength): # [cite: uploaded:spot_diagram.py]
        """Generates marginal rays at the stop at each pupil max,
        Px = (-1, +1), Py = (-1, +1)

        Returns:
            ray_tuple (tuple): Contains marginal each rays generated by
                trace_generic method, in north (Py=1), south (Py=-1),
                east (Px=1), west (Px=-1)directions.

        """
        ray_north = self.optic.trace_generic( # [cite: uploaded:spot_diagram.py]
            Hx=H_x, # [cite: uploaded:spot_diagram.py]
            Hy=H_y, # [cite: uploaded:spot_diagram.py]
            Px=0, # [cite: uploaded:spot_diagram.py]
            Py=1, # [cite: uploaded:spot_diagram.py]
            wavelength=wavelength, # [cite: uploaded:spot_diagram.py]
        )
        ray_south = self.optic.trace_generic( # [cite: uploaded:spot_diagram.py]
            Hx=H_x, # [cite: uploaded:spot_diagram.py]
            Hy=H_y, # [cite: uploaded:spot_diagram.py]
            Px=0, # [cite: uploaded:spot_diagram.py]
            Py=-1, # [cite: uploaded:spot_diagram.py]
            wavelength=wavelength, # [cite: uploaded:spot_diagram.py]
        )
        ray_east = self.optic.trace_generic( # [cite: uploaded:spot_diagram.py]
            Hx=H_x, # [cite: uploaded:spot_diagram.py]
            Hy=H_y, # [cite: uploaded:spot_diagram.py]
            Px=1, # [cite: uploaded:spot_diagram.py]
            Py=0, # [cite: uploaded:spot_diagram.py]
            wavelength=wavelength, # [cite: uploaded:spot_diagram.py]
        )
        ray_west = self.optic.trace_generic( # [cite: uploaded:spot_diagram.py]
            Hx=H_x, # [cite: uploaded:spot_diagram.py]
            Hy=H_y, # [cite: uploaded:spot_diagram.py]
            Px=-1, # [cite: uploaded:spot_diagram.py]
            Py=0, # [cite: uploaded:spot_diagram.py]
            wavelength=wavelength, # [cite: uploaded:spot_diagram.py]
        )

        ray_tuple = ray_north, ray_south, ray_east, ray_west # [cite: uploaded:spot_diagram.py]

        return ray_tuple # [cite: uploaded:spot_diagram.py]

    def generate_marginal_rays_cosines(self, H_x, H_y, wavelength): # [cite: uploaded:spot_diagram.py]
        """Generates directional cosines for each marginal ray. Calculates one
        field at a time (one height at a time).

        Returns:
            cosines_tuple (tuple): Contains directional cosines of marginal
                each rays.

        """
        ray_north, ray_south, ray_east, ray_west = self.generate_marginal_rays( # [cite: uploaded:spot_diagram.py]
            H_x, # [cite: uploaded:spot_diagram.py]
            H_y, # [cite: uploaded:spot_diagram.py]
            wavelength, # [cite: uploaded:spot_diagram.py]
        )

        north_cosines = be.array([ray_north.L, ray_north.M, ray_north.N]).ravel() # [cite: uploaded:spot_diagram.py]
        south_cosines = be.array([ray_south.L, ray_south.M, ray_south.N]).ravel() # [cite: uploaded:spot_diagram.py]
        east_cosines = be.array([ray_east.L, ray_east.M, ray_east.N]).ravel() # [cite: uploaded:spot_diagram.py]
        west_cosines = be.array([ray_west.L, ray_west.M, ray_west.N]).ravel() # [cite: uploaded:spot_diagram.py]

        cosines_tuple = (north_cosines, south_cosines, east_cosines, west_cosines) # [cite: uploaded:spot_diagram.py]
        return cosines_tuple # [cite: uploaded:spot_diagram.py]

    def generate_chief_rays_cosines(self, wavelength): # [cite: uploaded:spot_diagram.py]
        """Generates directional cosines for all chief rays of each field.

        Returns:
            chief_ray_cosines_list (ndarray): Mx3 numpy array, containing
                M number of directional cosines for all axes (x, y ,z).
            [
            field 1 -> (ray_chief1.L, ray_chief1.M, ray_chief1.N),
            field 2 -> (ray_chief1.L, ray_chief1.M, ray_chief1.N),
                .               .
                .               .
                .               .
            field M -> (ray_chiefM.L, ray_chiefM.M, ray_chiefM.N)
            ]

        """
        coords = self.fields # [cite: uploaded:spot_diagram.py]
        chief_ray_cosines_list = [] # [cite: uploaded:spot_diagram.py]
        for H_x, H_y in coords: # [cite: uploaded:spot_diagram.py]
            # Always pass the field values—even for 'angle' type.
            ray_chief = self.optic.trace_generic( # [cite: uploaded:spot_diagram.py]
                Hx=H_x, # [cite: uploaded:spot_diagram.py]
                Hy=H_y, # [cite: uploaded:spot_diagram.py]
                Px=0, # [cite: uploaded:spot_diagram.py]
                Py=0, # [cite: uploaded:spot_diagram.py]
                wavelength=wavelength, # [cite: uploaded:spot_diagram.py]
            )
            chief_ray_cosines_list.append( # [cite: uploaded:spot_diagram.py]
                be.array([ray_chief.L, ray_chief.M, ray_chief.N]).ravel(), # [cite: uploaded:spot_diagram.py]
            )
        chief_ray_cosines_list = be.stack(chief_ray_cosines_list, axis=0) # [cite: uploaded:spot_diagram.py]
        return chief_ray_cosines_list # [cite: uploaded:spot_diagram.py]

    def generate_chief_rays_centers(self, wavelength): # [cite: uploaded:spot_diagram.py]
        """Generates the position of each chief ray. It is used to find
        centers of the airy discs in the function _plot_field.

        Returns:
            chief_ray_centers (ndarray): Contains the x, y coordinates of
                chief ray centers for each field.

        """
        coords = self.fields # [cite: uploaded:spot_diagram.py]
        chief_ray_centers = [] # [cite: uploaded:spot_diagram.py]
        for H_x, H_y in coords: # [cite: uploaded:spot_diagram.py]
            # Always pass the field values—even for 'angle' type.
            ray_chief = self.optic.trace_generic( # [cite: uploaded:spot_diagram.py]
                Hx=H_x, # [cite: uploaded:spot_diagram.py]
                Hy=H_y, # [cite: uploaded:spot_diagram.py]
                Px=0, # [cite: uploaded:spot_diagram.py]
                Py=0, # [cite: uploaded:spot_diagram.py]
                wavelength=wavelength, # [cite: uploaded:spot_diagram.py]
            )
            x, y = ray_chief.x, ray_chief.y # [cite: uploaded:spot_diagram.py]
            chief_ray_centers.append([x, y]) # [cite: uploaded:spot_diagram.py]

        chief_ray_centers = be.stack(chief_ray_centers, axis=0) # [cite: uploaded:spot_diagram.py]
        return chief_ray_centers # [cite: uploaded:spot_diagram.py]

    def airy_disc_x_y(self, wavelength): # [cite: uploaded:spot_diagram.py]
        """Generates the airy radius for each x-y axes, for each field.

        Returns:
            airy_rad_tuple (tuple): Contains x axis centers (ndarray), and y
                axis centers (ndarray).

        """
        chief_ray_cosines_list = self.generate_chief_rays_cosines(wavelength) # [cite: uploaded:spot_diagram.py]

        airy_rad_x_list = [] # [cite: uploaded:spot_diagram.py]
        airy_rad_y_list = [] # [cite: uploaded:spot_diagram.py]

        for idx, (H_x, H_y) in enumerate(self.fields): # [cite: uploaded:spot_diagram.py]
            # Get marginal rays for the current field
            north_cos, south_cos, east_cos, west_cos = ( # [cite: uploaded:spot_diagram.py]
                self.generate_marginal_rays_cosines(H_x, H_y, wavelength) # [cite: uploaded:spot_diagram.py]
            )

            chief_cos = chief_ray_cosines_list[idx] # [cite: uploaded:spot_diagram.py]

            # Compute relative angles along x and y axes
            rel_angle_north = self.angle_from_cosine(chief_cos, north_cos) # [cite: uploaded:spot_diagram.py]
            rel_angle_south = self.angle_from_cosine(chief_cos, south_cos) # [cite: uploaded:spot_diagram.py]
            rel_angle_east = self.angle_from_cosine(chief_cos, east_cos) # [cite: uploaded:spot_diagram.py]
            rel_angle_west = self.angle_from_cosine(chief_cos, west_cos) # [cite: uploaded:spot_diagram.py]

            avg_angle_x = (rel_angle_north + rel_angle_south) / 2 # [cite: uploaded:spot_diagram.py]
            avg_angle_y = (rel_angle_east + rel_angle_west) / 2 # [cite: uploaded:spot_diagram.py]

            N_w_x = self.f_number(n=1, theta=avg_angle_x) # [cite: uploaded:spot_diagram.py]
            N_w_y = self.f_number(n=1, theta=avg_angle_y) # [cite: uploaded:spot_diagram.py]

            airy_rad_x = self.airy_radius(N_w_x, wavelength) # [cite: uploaded:spot_diagram.py]
            airy_rad_y = self.airy_radius(N_w_y, wavelength) # [cite: uploaded:spot_diagram.py]

            # convert to um
            airy_rad_x_list.append(airy_rad_x * 1e-3) # [cite: uploaded:spot_diagram.py]
            airy_rad_y_list.append(airy_rad_y * 1e-3) # [cite: uploaded:spot_diagram.py]

        airy_rad_tuple = (airy_rad_x_list, airy_rad_y_list) # [cite: uploaded:spot_diagram.py]

        return airy_rad_tuple # [cite: uploaded:spot_diagram.py]

    def centroid(self): # [cite: uploaded:spot_diagram.py]
        """Centroid of each spot

        Returns:
            centroid (List): centroid for each field in the data.

        """
        norm_index = self.optic.wavelengths.primary_index # [cite: uploaded:spot_diagram.py]
        if norm_index is None and self.optic.wavelengths.num_wavelengths > 0: # Fallback if no primary
            norm_index = 0
        elif norm_index is None: # No wavelengths at all
             return []


        centroid = [] # [cite: uploaded:spot_diagram.py]
        for field_data in self.data: # [cite: uploaded:spot_diagram.py]
            if not field_data or norm_index >= len(field_data): # Check if field_data is empty or norm_index is out of bounds
                centroid.append((0.0, 0.0)) # Default centroid or handle error
                continue
            spot_data_item = field_data[norm_index] # [cite: uploaded:spot_diagram.py]
            centroid_x = be.mean(spot_data_item.x) # [cite: uploaded:spot_diagram.py]
            centroid_y = be.mean(spot_data_item.y) # [cite: uploaded:spot_diagram.py]
            centroid.append((centroid_x, centroid_y)) # [cite: uploaded:spot_diagram.py]
        return centroid # [cite: uploaded:spot_diagram.py]

    def geometric_spot_radius(self): # [cite: uploaded:spot_diagram.py]
        """Geometric spot radius of each spot

        Returns:
            geometric_size (List): Geometric spot radius for field and
                wavelength

        """
        data = self._center_spots(self.data) # [cite: uploaded:spot_diagram.py]
        geometric_size = [] # [cite: uploaded:spot_diagram.py]
        for field_data in data: # [cite: uploaded:spot_diagram.py]
            geometric_size_field = [] # [cite: uploaded:spot_diagram.py]
            for wave_data in field_data: # [cite: uploaded:spot_diagram.py]
                r = be.sqrt(wave_data.x**2 + wave_data.y**2) # [cite: uploaded:spot_diagram.py]
                geometric_size_field.append(be.max(r)) # [cite: uploaded:spot_diagram.py]
            geometric_size.append(geometric_size_field) # [cite: uploaded:spot_diagram.py]
        return geometric_size # [cite: uploaded:spot_diagram.py]

    def rms_spot_radius(self): # [cite: uploaded:spot_diagram.py]
        """Root mean square (RMS) spot radius of each spot

        Returns:
            rms (List): RMS spot radius for each field and wavelength.

        """
        data = self._center_spots(self.data) # [cite: uploaded:spot_diagram.py]
        rms = [] # [cite: uploaded:spot_diagram.py]
        for field_data in data: # [cite: uploaded:spot_diagram.py]
            rms_field = [] # [cite: uploaded:spot_diagram.py]
            for wave_data in field_data: # [cite: uploaded:spot_diagram.py]
                r2 = wave_data.x**2 + wave_data.y**2 # [cite: uploaded:spot_diagram.py]
                rms_field.append(be.sqrt(be.mean(r2))) # [cite: uploaded:spot_diagram.py]
            rms.append(rms_field) # [cite: uploaded:spot_diagram.py]
        return rms # [cite: uploaded:spot_diagram.py]

    def _center_spots(self, data): # [cite: uploaded:spot_diagram.py]
        """Centers the spots in the given data around their respective centroids.

        Args:
            data (List): A nested list representing the data containing spots.

        Returns:
            data (List): A nested list with the spots centered around their
                centroids.

        """
        centroids = self.centroid() # [cite: uploaded:spot_diagram.py]
        if not centroids: # Handle case where centroids could not be calculated
            return data 

        centered = [] # [cite: uploaded:spot_diagram.py]
        for i, field_data_list in enumerate(data): # [cite: uploaded:spot_diagram.py]
            if i >= len(centroids): # Safety check
                centered.append(field_data_list) # Append original if no corresponding centroid
                continue
            field_copy_list = [] # [cite: uploaded:spot_diagram.py]
            for spot_data_item in field_data_list: # [cite: uploaded:spot_diagram.py]
                x2 = be.copy(spot_data_item.x) # [cite: uploaded:spot_diagram.py]
                y2 = be.copy(spot_data_item.y) # [cite: uploaded:spot_diagram.py]
                i2 = be.copy(spot_data_item.intensity) # [cite: uploaded:spot_diagram.py]

                x2 = x2 - centroids[i][0] # [cite: uploaded:spot_diagram.py]
                y2 = y2 - centroids[i][1] # [cite: uploaded:spot_diagram.py]

                field_copy_list.append(SpotData(x=x2, y=y2, intensity=i2)) # [cite: uploaded:spot_diagram.py]
            centered.append(field_copy_list) # [cite: uploaded:spot_diagram.py]
        return centered # [cite: uploaded:spot_diagram.py]

    def _generate_data( # [cite: uploaded:spot_diagram.py]
        self,
        fields,
        wavelengths,
        num_rays=100, # Renamed from num_rings to match trace_generic, though it's effectively number of rings for hexapolar
        distribution="hexapolar",
        coordinates="local",
    ):
        """Generate spot data for the given fields and wavelengths.

        Args:
            fields (List): A list of fields.
            wavelengths (List): A list of wavelengths.
            num_rays (int, optional): The number of rays to generate (used as num_rings for hexapolar).
                Defaults to 100.
            distribution (str, optional): The distribution type.
                Defaults to 'hexapolar'.
            coordinates (str): The coordinate system ('local' or 'global').

        Returns:
            data (List): A nested list of spot intersection data for each
                field and wavelength.

        """
        data = [] # [cite: uploaded:spot_diagram.py]
        for field in fields: # [cite: uploaded:spot_diagram.py]
            field_data = [] # [cite: uploaded:spot_diagram.py]
            for wavelength in wavelengths: # [cite: uploaded:spot_diagram.py]
                field_data.append( # [cite: uploaded:spot_diagram.py]
                    self._generate_field_data( # [cite: uploaded:spot_diagram.py]
                        field, wavelength, num_rays, distribution, coordinates
                    ),
                )
            data.append(field_data) # [cite: uploaded:spot_diagram.py]
        return data # [cite: uploaded:spot_diagram.py]

    def _generate_field_data( # [cite: uploaded:spot_diagram.py]
        self,
        field,
        wavelength,
        num_rays=100,
        distribution="hexapolar",
        coordinates="local",
    ):
        """Generates spot data for a given field and wavelength.

        Args:
            field (tuple): Tuple containing the field coordinates in (x, y).
            wavelength (float): The wavelength of the field.
            num_rays (int, optional): The number of rays to generate.
                Defaults to 100.
            distribution (str, optional): The distribution pattern of the
                rays. Defaults to 'hexapolar'.
            coordinates (str): The coordinate system ('local' or 'global').

        Returns:
            SpotData: An object containing x, y, and intensity values
                of the generated spot data.

        """
        # Optiland's trace takes num_rays, which for hexapolar effectively means num_rings
        self.optic.trace(*field, wavelength, num_rays, distribution) # [cite: uploaded:spot_diagram.py]

        x_global = self.optic.surface_group.x[-1, :] # [cite: uploaded:spot_diagram.py]
        y_global = self.optic.surface_group.y[-1, :] # [cite: uploaded:spot_diagram.py]
        z_global = self.optic.surface_group.z[-1, :] # [cite: uploaded:spot_diagram.py]
        intensity = self.optic.surface_group.intensity[-1, :] # [cite: uploaded:spot_diagram.py]

        if coordinates == "local": # [cite: uploaded:spot_diagram.py]
            plot_x, plot_y, _ = transform( # [cite: uploaded:spot_diagram.py]
                x_global, y_global, z_global, self.optic.image_surface, is_global=True # [cite: uploaded:spot_diagram.py]
            )
        else:  # coordinates == "global" # [cite: uploaded:spot_diagram.py]
            plot_x = x_global # [cite: uploaded:spot_diagram.py]
            plot_y = y_global # [cite: uploaded:spot_diagram.py]

        return SpotData(x=plot_x, y=plot_y, intensity=intensity) # [cite: uploaded:spot_diagram.py]

    def _plot_field( # [cite: uploaded:spot_diagram.py]
        self,
        ax,
        field_data,
        field,
        axis_lim,
        wavelengths,
        buffer=1.05,
        add_airy_disk=False,
        field_index=None,
        airy_rad_x=None,
        airy_rad_y=None,
        real_centroid_x=None,
        real_centroid_y=None,
    ):
        """Plot the field data on the given axis.
        (Code from uploaded spot_diagram.py, with be.to_numpy ensured)
        """
        markers = ["o", "s", "^"] # [cite: uploaded:spot_diagram.py]
        for k, points in enumerate(field_data): # [cite: uploaded:spot_diagram.py]
            x = points.x # [cite: uploaded:spot_diagram.py]
            y = points.y # [cite: uploaded:spot_diagram.py]
            intensity = points.intensity # [cite: uploaded:spot_diagram.py]
            x_np = be.to_numpy(x) # [cite: uploaded:spot_diagram.py]
            y_np = be.to_numpy(y) # [cite: uploaded:spot_diagram.py]
            i_np = be.to_numpy(intensity) # [cite: uploaded:spot_diagram.py]
            mask = i_np != 0 # [cite: uploaded:spot_diagram.py]
            ax.scatter( # [cite: uploaded:spot_diagram.py]
                x_np[mask], # [cite: uploaded:spot_diagram.py]
                y_np[mask], # [cite: uploaded:spot_diagram.py]
                s=10, # [cite: uploaded:spot_diagram.py]
                label=f"{wavelengths[k]:.4f} µm" if k < len(wavelengths) else "Unknown λ", # [cite: uploaded:spot_diagram.py]
                marker=markers[k % 3], # [cite: uploaded:spot_diagram.py]
                alpha=0.7, # [cite: uploaded:spot_diagram.py]
            )

        if add_airy_disk and field_index is not None and airy_rad_x is not None and airy_rad_y is not None and real_centroid_x is not None and real_centroid_y is not None: # [cite: uploaded:spot_diagram.py]
            real_centroid_x_np = be.to_numpy(real_centroid_x) # [cite: uploaded:spot_diagram.py]
            real_centroid_y_np = be.to_numpy(real_centroid_y) # [cite: uploaded:spot_diagram.py]
            airy_rad_x_np = be.to_numpy(airy_rad_x) # [cite: uploaded:spot_diagram.py]
            airy_rad_y_np = be.to_numpy(airy_rad_y) # [cite: uploaded:spot_diagram.py]
            if field_index < len(airy_rad_x_np) and field_index < len(airy_rad_y_np):
                ellipse = patches.Ellipse( # [cite: uploaded:spot_diagram.py]
                    (real_centroid_x_np, real_centroid_y_np), # [cite: uploaded:spot_diagram.py]
                    width=2 * airy_rad_y_np[field_index],  # diameter, not radius # [cite: uploaded:spot_diagram.py]
                    height=2 * airy_rad_x_np[field_index], # [cite: uploaded:spot_diagram.py]
                    linestyle="--", # [cite: uploaded:spot_diagram.py]
                    edgecolor="black", # [cite: uploaded:spot_diagram.py]
                    fill=False, # [cite: uploaded:spot_diagram.py]
                    linewidth=2, # [cite: uploaded:spot_diagram.py]
                )
                ax.add_patch(ellipse) # [cite: uploaded:spot_diagram.py]

                offset = abs(max(be.to_numpy(real_centroid_x), be.to_numpy(real_centroid_y))) # [cite: uploaded:spot_diagram.py] # Corrected to use be.to_numpy
                max_airy_x_val = max(airy_rad_x_np) if len(airy_rad_x_np) > 0 else 0 # [cite: uploaded:spot_diagram.py]
                max_airy_y_val = max(airy_rad_y_np) if len(airy_rad_y_np) > 0 else 0 # [cite: uploaded:spot_diagram.py]
                max_extent = max(axis_lim, max_airy_x_val, max_airy_y_val) # [cite: uploaded:spot_diagram.py]
                adjusted_axis_lim = (offset + max_extent) * buffer # [cite: uploaded:spot_diagram.py]
            else:
                 adjusted_axis_lim = axis_lim * buffer # [cite: uploaded:spot_diagram.py]
        else: # [cite: uploaded:spot_diagram.py]
            adjusted_axis_lim = axis_lim * buffer # [cite: uploaded:spot_diagram.py]

        cs = self.optic.image_surface.geometry.cs # [cite: uploaded:spot_diagram.py]
        effective_orientation = np.abs(cs.get_effective_rotation_euler()) # [cite: uploaded:spot_diagram.py]
        tol = 0.01  # [cite: uploaded:spot_diagram.py]
        if effective_orientation[0] > tol or effective_orientation[1] > tol: # [cite: uploaded:spot_diagram.py]
            x_label, y_label = "U (mm)", "V (mm)" # [cite: uploaded:spot_diagram.py]
        else: # [cite: uploaded:spot_diagram.py]
            x_label, y_label = "X (mm)", "Y (mm)" # [cite: uploaded:spot_diagram.py]

        ax.axis("square") # [cite: uploaded:spot_diagram.py]
        ax.set_xlabel(x_label) # [cite: uploaded:spot_diagram.py]
        ax.set_ylabel(y_label) # [cite: uploaded:spot_diagram.py]
        if adjusted_axis_lim > 1e-9: # Avoid setting tiny limits if axis_lim was near zero
            ax.set_xlim((-adjusted_axis_lim, adjusted_axis_lim)) # [cite: uploaded:spot_diagram.py]
            ax.set_ylim((-adjusted_axis_lim, adjusted_axis_lim)) # [cite: uploaded:spot_diagram.py]
        ax.set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}") # [cite: uploaded:spot_diagram.py]
        ax.grid(alpha=0.25) # [cite: uploaded:spot_diagram.py]

