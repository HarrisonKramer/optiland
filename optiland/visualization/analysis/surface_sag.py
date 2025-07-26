"""
SurgaceSagViewer

This module provides a viewer for visualizing the sag of an optical surface.
It generates a 2D sag map and 1D sag profiles along user-specified cross-sections.

Manuel Fragata Mendes, june 2025
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import optiland.backend as be
from optiland.visualization.base import BaseViewer


class SurfaceSagViewer(BaseViewer):
    """A viewer for visualizing the sag of an optical surface."""

    def view(
        self,
        surface_index: int,
        y_cross_section: float = 0.0,
        x_cross_section: float = 0.0,
    ):
        """
        Analyzes and visualizes the sag of a given lens surface.

        This method generates a 2D sag map and 1D sag profiles along
        user-specified x and y cross-sections.

        Args:
            surface_index (int): The index of the surface to analyze.
            y_cross_section (float, optional): The y-coordinate for the
                horizontal (X-axis) profile. Defaults to 0.0.
            x_cross_section (float, optional): The x-coordinate for the
                vertical (Y-axis) profile. Defaults to 0.0.
        """
        # For a 2D sag map, create grids for x and y
        num_points_grid = 50
        max_extent_grid = 5

        x_grid_coords = be.linspace(-max_extent_grid, max_extent_grid, num_points_grid)
        y_grid_coords = be.linspace(-max_extent_grid, max_extent_grid, num_points_grid)
        X_grid_map, Y_grid_map = be.meshgrid(x_grid_coords, y_grid_coords)

        surface = self.optic.surface_group.surfaces[surface_index]
        sag_map_2d = surface.geometry.sag(X_grid_map, Y_grid_map)

        # Calculate sag profiles at the specified cross-sections
        sag_profile_x = surface.geometry.sag(x_grid_coords, y_cross_section)
        sag_profile_y = surface.geometry.sag(x_cross_section, y_grid_coords)

        # --- Plotting using make_axes_locatable for robust alignment ---
        fig = plt.figure(figsize=(9, 9))
        ax_map = fig.add_subplot(111)

        # Create a divider for the main axes
        divider = make_axes_locatable(ax_map)

        # Append axes for the profiles and colorbar
        ax_profile_x = divider.append_axes("bottom", size="25%", pad=0.2, sharex=ax_map)
        ax_profile_y = divider.append_axes("right", size="25%", pad=0.2, sharey=ax_map)
        cax = divider.append_axes("top", size="5%", pad=0.1)

        # Hide labels on the shared axes
        ax_map.tick_params(axis="x", labelbottom=False)
        ax_profile_y.tick_params(axis="y", labelleft=False)

        # -- Main Plot: 2D Sag Map --
        ax_map.set_aspect("equal")
        contour = ax_map.contourf(
            be.to_numpy(X_grid_map),
            be.to_numpy(Y_grid_map),
            be.to_numpy(sag_map_2d),
            levels=50,
        )
        ax_map.set_ylabel("Y-coordinate")
        ax_map.set_title(f"2D Sag Map for Surface S{surface_index}", pad=80)

        # Add profile indicator lines at specified cross-sections
        ax_map.axhline(
            y_cross_section,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"X-Profile (y={y_cross_section})",
        )
        ax_map.axvline(
            x_cross_section,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label=f"Y-Profile (x={x_cross_section})",
        )
        ax_map.legend(loc="upper right")

        # -- Top: Colorbar --
        cbar = fig.colorbar(contour, cax=cax, orientation="horizontal")
        cbar.set_label("Sag (z)", labelpad=15, y=0.5)  # Adjust labelpad as needed
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

        # -- Bottom Plot: 1D X-Axis Sag Profile --
        ax_profile_x.plot(
            be.to_numpy(x_grid_coords),
            be.to_numpy(sag_profile_x),
            color="red",
            linestyle="--",
            linewidth=2,
        )
        ax_profile_x.set_xlabel("X-coordinate")
        ax_profile_x.set_ylabel("Sag (z)")
        ax_profile_x.grid(True)
        ax_profile_x.autoscale(enable=True, axis="y", tight=True)

        # -- Right Plot: 1D Y-Axis Sag Profile --
        ax_profile_y.plot(
            be.to_numpy(sag_profile_y),
            be.to_numpy(y_grid_coords),
            color="blue",
            linestyle="--",
            linewidth=2,
        )
        ax_profile_y.set_xlabel("Sag (z)")
        ax_profile_y.grid(True)
        ax_profile_y.autoscale(enable=True, axis="x", tight=True)

        fig.tight_layout(pad=1.0)
        plt.show()
