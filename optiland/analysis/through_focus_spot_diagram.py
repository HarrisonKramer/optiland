"""Through Focus Spot Diagram Analysis Module.

This module provides a class for performing through-focus spot diagram
analysis, calculating the spot diagram at various focal planes.

"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.analysis.spot_diagram import SpotDiagram
from optiland.analysis.through_focus import ThroughFocusAnalysis


class ThroughFocusSpotDiagram(ThroughFocusAnalysis):
    """Performs spot diagram analysis over a range of focal planes.

    This class extends `ThroughFocusAnalysis` to specifically calculate and
    report RMS spot radii from spot diagrams at various focal positions.
    It utilizes the `SpotDiagram` class for the core calculations at each
    focal plane.

    Attributes:
        optic (optiland.optic.Optic): The optical system being analyzed.
        delta_focus (float): The focal shift increment in mm.
        num_steps (int): Number of focal planes analyzed before and after
            the nominal focus.
        fields (list): Resolved list of field coordinates for analysis.
        wavelengths (list): Resolved list of wavelengths for analysis.
        num_rings (int): Number of rings for pupil sampling in the
            `SpotDiagram` calculation.
        distribution (str): Pupil sampling distribution type (e.g.,
            'hexapolar', 'random') for `SpotDiagram`.
        coordinates (Literal["global", "local"]): Coordinate system used for
            spot data generation within `SpotDiagram`.
        results (list[dict[float, list[float]]]): A list where each item is a
            dictionary. Each dictionary corresponds to a single focal plane
            and maps the delta focus (float, in mm) to a list of RMS spot
            radii (list of floats, in mm). Each RMS spot radius in the list
            corresponds to a field defined in `self.fields`, calculated at the
            primary wavelength.
    """

    def __init__(
        self,
        optic,
        delta_focus: float = 0.1,
        num_steps: int = 5,
        fields="all",
        wavelengths="all",
        num_rings: int = 6,
        distribution: str = "hexapolar",
        coordinates: Literal["global", "local"] = "local",
    ):
        """Initializes the ThroughFocusSpotDiagram analysis.

        Args:
            optic (optiland.optic.Optic): The optical system to analyze.
            delta_focus (float, optional): The increment of focal shift in mm.
                Defaults to 0.1.
            num_steps (int, optional): The number of focal planes to analyze
                on either side of the nominal focus. Defaults to 5. Must be in
                range [1, 7].
            fields (list[tuple[float,float]] | str, optional): Fields for
                analysis. If "all", uses all fields from `optic.fields`.
                Otherwise, expects a list of field coordinates.
                Defaults to "all".
            wavelengths (list[float] | str, optional): Wavelengths for
                analysis. If "all", uses all wavelengths from
                `optic.wavelengths`. Otherwise, expects a list of
                wavelength values. Defaults to "all".
            num_rings (int, optional): Number of rings for pupil sampling in
                the `SpotDiagram` calculation. Defaults to 6.
            distribution (str, optional): Pupil sampling distribution type for
                `SpotDiagram` (e.g., 'hexapolar', 'random').
                Defaults to "hexapolar".
            coordinates (Literal["global", "local"], optional): Coordinate
                system for spot data generation in `SpotDiagram`.
                Defaults to "local".
        """
        self.num_rings = num_rings
        self.distribution = distribution
        if coordinates not in ["global", "local"]:
            raise ValueError("Coordinates must be 'global' or 'local'.")
        self.coordinates = coordinates

        super().__init__(
            optic,
            delta_focus=delta_focus,
            num_steps=num_steps,
            fields=fields,
            wavelengths=wavelengths,
        )

    def _perform_analysis_at_focus(self):
        """Calculates RMS spot radii at the current focal plane.

        This method is called by the base class for each focal step. It
        instantiates a `SpotDiagram` object for the optic's current focal
        state, calculates the RMS spot radius for each specified field at the
        primary wavelength, and returns this data.

        Note:
            This implementation re-instantiates `SpotDiagram` for each focal
            step, which involves recalculating ray data. For high-performance
            needs, optimizing this by directly accessing or reusing ray tracing
            functionality might be considered.

        Returns:
            list: a list of spot diagram data, including intersection points and
                intensity
        """
        spot_diagram_at_focus = SpotDiagram(
            self.optic,
            fields=self.fields,
            wavelengths=self.wavelengths,
            num_rings=self.num_rings,
            distribution=self.distribution,
            coordinates=self.coordinates,
        )
        return spot_diagram_at_focus.data

    def view(self, figsize_per_plot=(3, 3), buffer=1.05):
        """Visualizes the through-focus spot diagrams.

        Generates a grid of plots where rows represent fields and columns
        represent defocus positions. Each plot shows the spot diagram for all
        wavelengths, centered by its primary wavelength centroid.

        Args:
            figsize_per_plot (tuple, optional): Approximate (width, height)
                in inches for each individual subplot. Defaults to (3,3).
            buffer (float, optional): Buffer factor to extend the axis limits
                beyond the maximum spot extent. Default is 1.05.
        """
        if not self.results:
            print("No data to display. Run analysis first.")
            return

        num_fields = len(self.fields)
        num_defocus_steps = self.num_steps

        if num_fields == 0 or num_defocus_steps == 0 or not self.wavelengths:
            print("No fields, defocus steps, or wavelengths to plot.")
            return

        # Calculate global axis limit based on max geometric radius after centering
        max_r_sq_numpy = 0.0
        for j_defocus_idx in range(num_defocus_steps):
            data_this_defocus = self.results[j_defocus_idx]
            for i_field_idx in range(num_fields):
                if not data_this_defocus[i_field_idx]:
                    continue  # Skip if no wavelength data for this field

                primary_wl_idx = self.optic.wavelengths.primary_index
                # Ensure primary_wl_idx is valid for the current field's wavelength data
                if primary_wl_idx >= len(data_this_defocus[i_field_idx]):
                    current_field_wl_data = data_this_defocus[i_field_idx][0]
                else:
                    current_field_wl_data = data_this_defocus[i_field_idx][
                        primary_wl_idx
                    ]

                if be.any(current_field_wl_data.intensity == 0) and be.all(
                    current_field_wl_data.intensity == 0
                ):
                    # skip if all intensities are zero, avoids nan in mean
                    plot_centroid_x_val, plot_centroid_y_val = 0.0, 0.0
                else:
                    plot_centroid_x = be.mean(
                        current_field_wl_data.x[current_field_wl_data.intensity != 0]
                    )
                    plot_centroid_y = be.mean(
                        current_field_wl_data.y[current_field_wl_data.intensity != 0]
                    )
                    plot_centroid_x_val = be.to_numpy(plot_centroid_x).item()
                    plot_centroid_y_val = be.to_numpy(plot_centroid_y).item()

                for k_wl_idx in range(len(self.wavelengths)):
                    spot_data_item = data_this_defocus[i_field_idx][k_wl_idx]

                    x_centered = spot_data_item.x - plot_centroid_x_val
                    y_centered = spot_data_item.y - plot_centroid_y_val

                    # Consider only valid rays (intensity != 0) for radius calculation
                    valid_rays_mask = spot_data_item.intensity != 0
                    if be.any(valid_rays_mask):
                        r_sq_values = (
                            x_centered[valid_rays_mask] ** 2
                            + y_centered[valid_rays_mask] ** 2
                        )
                        current_max_r_sq = be.max(r_sq_values)
                        max_r_sq_numpy = max(
                            max_r_sq_numpy, be.to_numpy(current_max_r_sq).item()
                        )

        global_axis_lim = (
            np.sqrt(max_r_sq_numpy) if max_r_sq_numpy > 0 else 0.01
        )  # Default small limit

        fig, axs = plt.subplots(
            num_fields,
            num_defocus_steps,
            figsize=(
                num_defocus_steps * figsize_per_plot[0],
                num_fields * figsize_per_plot[1],
            ),
            sharex=True,
            sharey=True,
            squeeze=False,  # Ensures axs is always 2D
        )

        markers = ["o", "s", "^"]
        legend_handles = []
        legend_labels = []

        # Determine X/Y labels based on image surface orientation (once)
        cs = self.optic.image_surface.geometry.cs
        effective_orientation = np.abs(be.to_numpy(cs.get_effective_rotation_euler()))
        tol = 0.01
        if effective_orientation[0] > tol or effective_orientation[1] > tol:
            x_plot_label, y_plot_label = "U (mm)", "V (mm)"
        else:
            x_plot_label, y_plot_label = "X (mm)", "Y (mm)"

        for i_field_idx in range(num_fields):
            for j_defocus_idx in range(num_defocus_steps):
                ax = axs[i_field_idx, j_defocus_idx]
                field_coord = self.fields[i_field_idx]

                # nominal_focus and positions[j] are numpy arrays from backend
                # Convert to float for subtraction if they are 0-d arrays
                nominal_focus_val = be.to_numpy(self.nominal_focus).item()
                position_val = float(self.positions[j_defocus_idx])
                defocus_val = position_val - nominal_focus_val

                data_for_this_plot = self.results[j_defocus_idx][i_field_idx]

                primary_wl_idx = self.optic.wavelengths.primary_index

                if primary_wl_idx >= len(data_for_this_plot):
                    spot_data_primary_wl_plot = data_for_this_plot[0]  # Fallback
                else:
                    spot_data_primary_wl_plot = data_for_this_plot[primary_wl_idx]

                if be.any(spot_data_primary_wl_plot.intensity == 0) and be.all(
                    spot_data_primary_wl_plot.intensity == 0
                ):
                    plot_centroid_x_val, plot_centroid_y_val = 0.0, 0.0
                else:
                    plot_centroid_x = be.mean(
                        spot_data_primary_wl_plot.x[
                            spot_data_primary_wl_plot.intensity != 0
                        ]
                    )
                    plot_centroid_y = be.mean(
                        spot_data_primary_wl_plot.y[
                            spot_data_primary_wl_plot.intensity != 0
                        ]
                    )
                    plot_centroid_x_val = be.to_numpy(plot_centroid_x).item()
                    plot_centroid_y_val = be.to_numpy(plot_centroid_y).item()

                for k_wl_idx, spot_data_item in enumerate(data_for_this_plot):
                    x_centered_plot = spot_data_item.x - plot_centroid_x_val
                    y_centered_plot = spot_data_item.y - plot_centroid_y_val

                    x_np = be.to_numpy(x_centered_plot)
                    y_np = be.to_numpy(y_centered_plot)
                    i_np = be.to_numpy(spot_data_item.intensity)
                    mask = i_np != 0

                    if be.any(mask):  # Only plot if there are any valid rays
                        scatter_plot = ax.scatter(
                            x_np[mask],
                            y_np[mask],
                            s=10,
                            marker=markers[k_wl_idx % len(markers)],
                            alpha=0.7,
                        )
                        if i_field_idx == 0 and j_defocus_idx == 0:
                            wl_value = self.wavelengths[k_wl_idx]
                            legend_handles.append(scatter_plot)
                            legend_labels.append(f"{wl_value:.4f} µm")

                ax.axis("square")
                ax.grid(alpha=0.25)

                # Titles and labels
                title_str = f"Field: ({field_coord[0]:.2f},{field_coord[1]:.2f})"
                if i_field_idx == 0:  # Add defocus to title for top row
                    title_str = f"Defocus: {defocus_val:+.3f} mm\n{title_str}"
                ax.set_title(title_str, fontsize=10)

                if i_field_idx == num_fields - 1:
                    ax.set_xlabel(x_plot_label)
                if j_defocus_idx == 0:
                    ax.set_ylabel(y_plot_label)

                current_lim = global_axis_lim * buffer
                ax.set_xlim((-current_lim, current_lim))
                ax.set_ylim((-current_lim, current_lim))

        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=min(len(legend_labels), 5),  # Max 5 columns for legend
                bbox_to_anchor=(0.5, -0.02 / (figsize_per_plot[1] * num_fields / 4)),
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust rect for legend and titles
        plt.show()
