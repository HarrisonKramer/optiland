"""Y Y-bar Analysis

This module provides a y y-bar analysis for optical systems.
This is a plot of the marginal ray height versus the chief ray height
for each surface in the system.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt

import optiland.backend as be

from .base import BaseAnalysis


class YYbar(BaseAnalysis):
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

    def __init__(self, optic, wavelength="primary"):
        if isinstance(wavelength, str) and wavelength.lower() == "primary":
            self.wavelength_value_for_display = optic.primary_wavelength
        elif isinstance(wavelength, float | int):
            self.wavelength_value_for_display = float(wavelength)
        else:
            self.wavelength_value_for_display = optic.primary_wavelength

        super().__init__(optic, wavelengths=[self.wavelength_value_for_display])

    def _generate_data(self):
        """
        Generates the data for the YY-bar plot by tracing paraxial marginal
        and chief rays.
        """
        try:
            # Paraxial rays are traced using the primary wavelength defined in the optic
            ya, _ = self.optic.paraxial.marginal_ray()
            yb, _ = self.optic.paraxial.chief_ray()
            return {"ya": ya.flatten(), "yb": yb.flatten()}
        except Exception as e:
            print(f"Error generating YYbar data: {e}")
            return None

    def view(
        self, fig_to_plot_on: plt.Figure = None, figsize: tuple[float, float] = (7, 5.5)
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Visualizes the ray heights of the marginal and chief rays in a Y Y-bar diagram.

        This method plots the relationship between the chief ray height (Y-bar) and
        the marginal ray height (Y)
        for each surface in the optical system. It can either plot on a provided
        matplotlib Figure (for GUI embedding)
        or create a new figure.

        Parameters
        ----------
        fig_to_plot_on : plt.Figure, optional
            An existing matplotlib Figure to plot on. If None, a new figure is created.
            Default is None.
        figsize : tuple of float, optional
            Size of the figure to create if `fig_to_plot_on` is None.
                Default is (7, 5.5).

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The matplotlib Figure and Axes objects containing the plot.

        Notes
        -----
        - If the required data (`ya` and `yb`) is not available, an error message is
        displayed on the plot.
        - Surface labels are generated based on surface comments, IDs, or their role
        (e.g., "Image", "Stop").
        - The plot includes horizontal and vertical reference lines at zero.
        - The diagram's title includes the wavelength used for the calculation.

        """
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        if self.data is None or "ya" not in self.data or "yb" not in self.data:
            ax.text(
                0.5,
                0.5,
                "Error: YY-bar data could not be generated.",
                ha="center",
                va="center",
                color="red",
            )
            if is_gui_embedding and hasattr(current_fig, "canvas"):
                current_fig.canvas.draw_idle()
            return

        ya = self.data["ya"]
        yb = self.data["yb"]
        num_surfaces_in_optic = self.optic.surface_group.num_surfaces

        # Plotting logic with corrected labeling
        for k in range(1, num_surfaces_in_optic):
            label = ""
            surface_obj_prev = self.optic.surface_group.surfaces[k - 1]

            if k == 1:
                label = (
                    surface_obj_prev.comment
                    if surface_obj_prev.comment and surface_obj_prev.comment != "Object"
                    else (
                        f"S{surface_obj_prev.id}"
                        if hasattr(surface_obj_prev, "id")
                        else f"S{k - 1}"
                    )
                )
                if k - 1 == self.optic.surface_group.stop_index:
                    label += " (Stop)"

            if k < num_surfaces_in_optic - 1:
                surface_obj_curr = self.optic.surface_group.surfaces[k]
                label = (
                    surface_obj_curr.comment
                    if surface_obj_curr.comment
                    else (
                        f"S{surface_obj_curr.id}"
                        if hasattr(surface_obj_curr, "id")
                        else f"S{k}"
                    )
                )
                if k == self.optic.surface_group.stop_index:
                    label += " (Stop)"
            elif k == num_surfaces_in_optic - 1:
                label = "Image"

            ax.plot(
                [be.to_numpy(yb[k - 1]), be.to_numpy(yb[k])],
                [be.to_numpy(ya[k - 1]), be.to_numpy(ya[k])],
                ".-",
                label=(
                    label
                    if k == 1
                    or k == num_surfaces_in_optic - 1
                    or k == self.optic.surface_group.stop_index
                    else None
                ),
                markersize=8,
            )

        ax.axhline(y=0, linewidth=0.5, color="k")
        ax.axvline(x=0, linewidth=0.5, color="k")
        ax.set_xlabel("Chief Ray Height (mm)")
        ax.set_ylabel("Marginal Ray Height (mm)")
        ax.set_title(f"Y Y-bar Diagram (λ={self.wavelength_value_for_display:.3f} µm)")
        ax.legend()
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        return current_fig, ax
