"""Y Y-bar Analysis

This module provides a y y-bar analysis for optical systems.
This is a plot of the marginal ray height versus the chief ray height
for each surface in the system.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt
import optiland.backend as be #


class YYbar:
    """Class representing the YYbar analysis of an optic.

    Args:
        optic (Optic): The optic object to analyze.
        wavelength (str or float, optional): The specific wavelength value (in µm)
            or the string 'primary' to use the optic's primary wavelength.
            Defaults to 'primary'. This is primarily for display in the plot title,
            as paraxial rays use the optic's current primary setting.

    Methods:
        view(fig_to_plot_on=None, figsize=(7, 5.5)): Visualizes the YYbar analysis.
            If fig_to_plot_on is provided, plots on that Matplotlib Figure.
            Otherwise, creates a new figure and shows it.
    """

    def __init__(self, optic, wavelength="primary"): #
        self.optic = optic #
        # Determine the wavelength value for reference (e.g., plot title)
        if isinstance(wavelength, str) and wavelength.lower() == "primary":
            if optic.wavelengths.primary_index is not None and \
               0 <= optic.wavelengths.primary_index < optic.wavelengths.num_wavelengths:
                self.wavelength_value = optic.wavelengths.wavelengths[optic.wavelengths.primary_index].value
            elif optic.wavelengths.num_wavelengths > 0: # Fallback to first wavelength
                self.wavelength_value = optic.wavelengths.wavelengths[0].value
                # Optionally, you might want to log a warning if primary_index was None but wavelengths exist
            else:
                # This case should be rare if Optic/OptilandConnector ensures a default wavelength
                print("Warning (YYbar.__init__): Optic has no wavelengths. Using a default 0.550 µm for title.")
                self.wavelength_value = 0.550
        elif isinstance(wavelength, (float, int)):
            self.wavelength_value = float(wavelength)
        else:
            print(f"Warning (YYbar.__init__): Unexpected wavelength format '{wavelength}'. Using primary or default.")
            # Fallback to primary or default if format is unexpected
            if optic.wavelengths.primary_index is not None and \
               0 <= optic.wavelengths.primary_index < optic.wavelengths.num_wavelengths:
                self.wavelength_value = optic.wavelengths.wavelengths[optic.wavelengths.primary_index].value
            elif optic.wavelengths.num_wavelengths > 0:
                self.wavelength_value = optic.wavelengths.wavelengths[0].value
            else:
                self.wavelength_value = 0.550


    def view(self, fig_to_plot_on=None, figsize=(7, 5.5)): #
        """Visualizes the ray heights of the marginal and chief rays.

        Args:
            fig_to_plot_on (matplotlib.figure.Figure, optional): The Matplotlib
                figure object to plot on. If None, a new figure is created.
            figsize (tuple): The size of the figure (width, height) if a new
                figure is created. Default is (7, 5.5).
        """
        is_gui_embedding = fig_to_plot_on is not None

        if not is_gui_embedding:
            current_fig = plt.figure(figsize=figsize) # Create new figure for standalone
            ax = current_fig.add_subplot(111) # Add a single subplot
        else:
            current_fig = fig_to_plot_on # Use GUI's figure
            current_fig.clear()          # Clear it for the new plot
            ax = current_fig.add_subplot(111) # Add a single subplot to the GUI's figure

        # Paraxial rays will be traced using the optic's current primary wavelength setting
        try:
            ya, _ = self.optic.paraxial.marginal_ray() #
            yb, _ = self.optic.paraxial.chief_ray() #
        except Exception as e:
            print(f"Error (YYbar.view): Failed to get paraxial rays from optic: {e}")
            ax.text(0.5, 0.5, f"Error fetching paraxial rays:\n{e}",
                    ha='center', va='center', transform=ax.transAxes, color='red', wrap=True)
            if hasattr(current_fig, 'canvas') and current_fig.canvas is not None:
                 current_fig.canvas.draw_idle()
            return

        ya = ya.flatten() #
        yb = yb.flatten() #

        # Determine labels based on surface comments or IDs
        # surface 0 = object, surface 1 = first optical surface, ..., last surface = image
        num_surfaces_in_optic = self.optic.surface_group.num_surfaces
        
        for k in range(1, num_surfaces_in_optic): # Iterate through optical surfaces up to image plane
                                                # k here refers to the index in ya, yb arrays after surface 0 (object)
                                                # ya[k] is ray height AT surface k of the Optic object
                                                # The segment is from surface k-1 to surface k.
            label = ""
            # Optic surface indices are 0 (obj), 1 (first lens surf), ..., N-1 (image)
            # Ray data arrays ya, yb might have N points (0 to N-1) corresponding to these N surfaces
            # Or N-1 segments (if they represent data *between* surfaces)
            # Assuming ya[k], yb[k] are heights AT surface k (0-indexed in optic.surface_group.surfaces)

            # Plot segment from previous surface (k-1) to current surface (k)
            # Heights at previous surface: ya[k-1], yb[k-1]
            # Heights at current surface: ya[k], yb[k]

            surface_obj_prev = self.optic.surface_group.surfaces[k-1]
            
            if k == 1: # First segment, from Object (surf 0) to First Optical Surface (surf 1)
                       # Label refers to the surface the ray segment is *approaching* or where it lands
                label = surface_obj_prev.comment if surface_obj_prev.comment and surface_obj_prev.comment != "Object" \
                        else (f"S{surface_obj_prev.id}" if hasattr(surface_obj_prev, 'id') else f"S{k-1}")
                if k-1 == self.optic.surface_group.stop_index:
                    label += " (Stop)"

            # Label for the surface where the segment ends (surface k)
            if k < num_surfaces_in_optic -1 : # For intermediate optical surfaces
                surface_obj_curr = self.optic.surface_group.surfaces[k]
                label = surface_obj_curr.comment if surface_obj_curr.comment else \
                        (f"S{surface_obj_curr.id}" if hasattr(surface_obj_curr, 'id') else f"S{k}")
                if k == self.optic.surface_group.stop_index:
                    label += " (Stop)"
            elif k == num_surfaces_in_optic - 1: # Last segment, ending at Image surface
                label = "Image"


            ax.plot(
                [be.to_numpy(yb[k - 1]), be.to_numpy(yb[k])], #
                [be.to_numpy(ya[k - 1]), be.to_numpy(ya[k])], #
                ".-", #
                label=label if k==1 or k == num_surfaces_in_optic - 1 or k == self.optic.surface_group.stop_index else None, # Only label key surfaces for clarity
                markersize=8, #
            )

        ax.axhline(y=0, linewidth=0.5, color="k") #
        ax.axvline(x=0, linewidth=0.5, color="k") #
        ax.set_xlabel("Chief Ray Height (mm)") #
        ax.set_ylabel("Marginal Ray Height (mm)") #
        ax.set_title(f"Y Y-bar Diagram (λ={self.wavelength_value:.3f} µm)")
        
        handles, labels = ax.get_legend_handles_labels()
        if handles: 
            ax.legend()

        current_fig.tight_layout() # Adjust layout to prevent overlap/truncation

        if not is_gui_embedding:
            plt.show() #
        else:
            # The GUI's canvas (current_fig.canvas) will be redrawn by the GUI
            if hasattr(current_fig, 'canvas') and current_fig.canvas is not None:
                current_fig.canvas.draw_idle()
