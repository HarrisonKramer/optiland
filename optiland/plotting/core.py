"""Core plotting functionalities for OptiLand."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np # For example usage

try:
    from .config import PlotConfig
except ImportError:
    # This allows the script to be run directly for testing,
    # assuming config.py is in the same directory.
    from config import PlotConfig

class Plotter:
    """
    A class to create styled plots using Matplotlib, configured by PlotConfig.
    """

    def __init__(self, config=None):
        """
        Initializes the Plotter with a PlotConfig instance.

        Args:
            config (PlotConfig, optional): A PlotConfig instance.
                                           If None, a default PlotConfig is created.
        """
        self.config = config if config is not None else PlotConfig()

    def create_figure_and_axes(self, figsize=(10, 4), **kwargs):
        """
        Creates a Matplotlib figure and axes, styled according to the current theme.

        Args:
            figsize (tuple, optional): The size of the figure (width, height) in inches.
                                       Defaults to (10, 4).
            **kwargs: Additional keyword arguments to pass to `plt.subplots()`.

        Returns:
            tuple: A tuple containing the Matplotlib figure and axes (fig, ax).
        """
        fig, ax = plt.subplots(figsize=figsize, **kwargs)

        fig.set_facecolor(self.config.get_style("background_color"))
        ax.set_facecolor(self.config.get_style("background_color"))

        axis_color = self.config.get_style("axis_color")
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_color(axis_color)

        ax.tick_params(axis='x', colors=axis_color)
        ax.tick_params(axis='y', colors=axis_color)

        return fig, ax

    def plot_lines(self, ax, z_coords, y_coords_list, style_key_prefix="line_colors", linewidth=1):
        """
        Plots multiple lines on the given axes.

        Colors are cycled from the theme's line_colors.

        Args:
            ax (matplotlib.axes.Axes): The axes to plot on.
            z_coords (array-like): The x-coordinates for the plot.
            y_coords_list (list of array-like): A list of y-coordinate sets.
            style_key_prefix (str, optional): The prefix for line color keys in config.
                                              Defaults to "line_colors".
            linewidth (float, optional): The width of the lines. Defaults to 1.
        """
        num_colors = len(self.config.get_style(style_key_prefix))
        for i, y_data in enumerate(y_coords_list):
            line_color = self.config.get_style(f"{style_key_prefix}.{i % num_colors}")
            ax.plot(z_coords, y_data, color=line_color, linewidth=linewidth, label=f'Line {i+1}')

    def add_polygon(self, ax, vertices, style_key="patch_colors.default", closed=True, **kwargs):
        """
        Adds a styled polygon to the axes.

        Args:
            ax (matplotlib.axes.Axes): The axes to add the polygon to.
            vertices (list of tuples): The vertices of the polygon.
            style_key (str, optional): The key for patch colors in config.
                                       Defaults to "patch_colors.default".
            closed (bool, optional): Whether the polygon is closed. Defaults to True.
            **kwargs: Additional keyword arguments to pass to `matplotlib.patches.Polygon()`.
        """
        facecolor = self.config.get_style(f"{style_key}.facecolor")
        edgecolor = self.config.get_style(f"{style_key}.edgecolor")

        polygon = patches.Polygon(vertices, closed=closed,
                                  facecolor=facecolor, edgecolor=edgecolor, **kwargs)
        ax.add_patch(polygon)
        return polygon

    def set_labels(self, ax, xlabel="Z [mm]", ylabel="Y [mm]"):
        """
        Sets the x and y axis labels with themed text color.

        Args:
            ax (matplotlib.axes.Axes): The axes to set labels for.
            xlabel (str, optional): The label for the x-axis. Defaults to "Z [mm]".
            ylabel (str, optional): The label for the y-axis. Defaults to "Y [mm]".
        """
        text_color = self.config.get_style("text_color")
        ax.set_xlabel(xlabel, color=text_color)
        ax.set_ylabel(ylabel, color=text_color)

    def set_title(self, ax, title):
        """
        Sets the title of the plot with themed text color.

        Args:
            ax (matplotlib.axes.Axes): The axes to set the title for.
            title (str): The title of the plot.
        """
        text_color = self.config.get_style("text_color")
        ax.set_title(title, color=text_color)

    def set_xlim(self, ax, xlim):
        """
        Sets the x-axis limits.

        Args:
            ax (matplotlib.axes.Axes): The axes to set limits for.
            xlim (tuple): The (min, max) limits for the x-axis.
        """
        ax.set_xlim(xlim)

    def set_ylim(self, ax, ylim):
        """
        Sets the y-axis limits.

        Args:
            ax (matplotlib.axes.Axes): The axes to set limits for.
            ylim (tuple): The (min, max) limits for the y-axis.
        """
        ax.set_ylim(ylim)

    def apply_grid(self, ax, style_key="grid_color", alpha=0.25, linestyle="--"):
        """
        Applies a styled grid to the axes.

        Args:
            ax (matplotlib.axes.Axes): The axes to apply the grid to.
            style_key (str, optional): The key for the grid color in config.
                                       Defaults to "grid_color".
            alpha (float, optional): Transparency of the grid. Defaults to 0.25.
            linestyle (str, optional): Linestyle of the grid. Defaults to "--".
        """
        grid_color = self.config.get_style(style_key)
        ax.grid(True, color=grid_color, alpha=alpha, linestyle=linestyle)

    def set_aspect_equal(self, ax):
        """
        Sets the aspect ratio of the plot to be equal.

        Args:
            ax (matplotlib.axes.Axes): The axes to set the aspect ratio for.
        """
        ax.set_aspect('equal')

    def show_plot(self):
        """Displays the current plot."""
        plt.show()

    def save_figure(self, fig, filename, dpi=300, **kwargs):
        """
        Saves the figure to a file.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            filename (str): The name of the file to save the figure to.
            dpi (int, optional): The resolution of the saved figure. Defaults to 300.
            **kwargs: Additional keyword arguments to pass to `fig.savefig()`.
        """
        # Ensure facecolor from config is used, as it might be overridden by theme styles
        fig.savefig(filename, dpi=dpi, facecolor=fig.get_facecolor(), **kwargs)
        print(f"Figure saved to {filename}")

    def add_legend(self, ax, **kwargs):
        """
        Adds a legend to the axes, styled with theme colors.

        Args:
            ax (matplotlib.axes.Axes): The axes to add the legend to.
            **kwargs: Additional keyword arguments to pass to `ax.legend()`.
        """
        legend = ax.legend(**kwargs)
        if legend: # legend can be None if no labeled artists are present
            text_color = self.config.get_style("text_color")
            for text in legend.get_texts():
                text.set_color(text_color)
            if legend.get_frame():
                legend.get_frame().set_facecolor(self.config.get_style("background_color"))
                legend.get_frame().set_edgecolor(self.config.get_style("axis_color"))


    def add_colorbar(self, fig, mappable, label="", **kwargs):
        """
        Adds a colorbar to the figure, styled with theme colors.

        Args:
            fig (matplotlib.figure.Figure): The figure to add the colorbar to.
            mappable (matplotlib.cm.ScalarMappable): The mappable object (e.g., an image or contour plot)
                                                     to which the colorbar applies.
            label (str, optional): The label for the colorbar. Defaults to "".
            **kwargs: Additional keyword arguments to pass to `fig.colorbar()`.

        Returns:
            matplotlib.colorbar.Colorbar: The created colorbar instance.
        """
        cb = fig.colorbar(mappable, label=label, **kwargs)

        text_color = self.config.get_style("text_color")
        axis_color = self.config.get_style("axis_color")

        cb.set_label(label, color=text_color)
        cb.ax.yaxis.set_tick_params(color=axis_color, labelcolor=text_color)
        cb.outline.set_edgecolor(axis_color)

        return cb

    def tight_layout(self, fig, **kwargs):
        """Applies fig.tight_layout()."""
        fig.tight_layout(**kwargs)

if __name__ == '__main__':
    # Example Usage:

    # Create a Plotter instance (uses "light" theme by default)
    plotter = Plotter()

    # 1. Test with light theme
    fig, ax = plotter.create_figure_and_axes()

    # Plot some lines
    z_coords = np.linspace(0, 10, 100)
    y_coords1 = np.sin(z_coords)
    y_coords2 = np.cos(z_coords)
    y_coords3 = np.sin(z_coords * 0.5) * 0.5
    plotter.plot_lines(ax, z_coords, [y_coords1, y_coords2, y_coords3])

    # Add a polygon
    poly_vertices = [(1, -0.5), (3, 0.5), (2, 0.8)]
    plotter.add_polygon(ax, poly_vertices, style_key="patch_colors.lens", alpha=0.5)

    default_poly_verts = [(5, -0.8), (7, -0.2), (6, -0.1)]
    plotter.add_polygon(ax, default_poly_verts, style_key="patch_colors.default", alpha=0.7)


    # Configure chart
    plotter.set_labels(ax, xlabel="X-axis (units)", ylabel="Y-axis (units)")
    plotter.set_title(ax, "Light Theme Example Plot")
    plotter.set_xlim(ax, (0, 10))
    plotter.set_ylim(ax, (-1.5, 1.5))
    plotter.apply_grid(ax)
    # plotter.set_aspect_equal(ax) # Uncomment to test aspect ratio
    plotter.add_legend(ax, loc='upper right')

    # Save the light theme plot
    # plotter.save_figure(fig, "light_theme_example.png")
    # plotter.show_plot() # Show plot if not saving, or if you want to see before saving
    plt.close(fig) # Close the figure to free memory before the next plot

    print("Light theme example generated.")

    # 2. Test with dark theme
    dark_config = PlotConfig(initial_theme="dark")
    plotter_dark = Plotter(config=dark_config)

    fig_dark, ax_dark = plotter_dark.create_figure_and_axes(figsize=(12,5))

    plotter_dark.plot_lines(ax_dark, z_coords, [y_coords1 * 0.8, y_coords2 * 0.8, y_coords3 * 0.8], linewidth=1.5)

    poly_vertices_dark = [(7, 0.0), (9, 1.0), (8, 1.2)]
    plotter_dark.add_polygon(ax_dark, poly_vertices_dark, style_key="patch_colors.lens", alpha=0.6)

    default_poly_verts_dark = [(1, 0.2), (3, 0.8), (2, 0.9)]
    plotter_dark.add_polygon(ax_dark, default_poly_verts_dark, style_key="patch_colors.default", alpha=0.7)

    plotter_dark.set_labels(ax_dark, xlabel="X Data", ylabel="Y Data")
    plotter_dark.set_title(ax_dark, "Dark Theme Example Plot")
    plotter_dark.set_xlim(ax_dark, (0, 10))
    plotter_dark.set_ylim(ax_dark, (-1.5, 1.5))
    plotter_dark.apply_grid(ax_dark, alpha=0.2, linestyle=':')
    plotter_dark.add_legend(ax_dark, loc='lower left')

    # Test colorbar
    try:
        # Create a dummy mappable for colorbar
        data = np.random.rand(10,10)
        mappable = ax_dark.imshow(data, cmap='viridis') # Viridis is just an example
        cb = plotter_dark.add_colorbar(fig_dark, mappable, label="Intensity")
        # print(f"Colorbar added. Label color: {cb.label.get_color()}") # Requires access to internal matplotlib state
    except Exception as e:
        print(f"Error adding colorbar in example: {e}")


    # Save the dark theme plot
    # plotter_dark.save_figure(fig_dark, "dark_theme_example.png")
    # plotter_dark.show_plot()
    plt.close(fig_dark)
    print("Dark theme example generated.")

    print("\nExample usage complete. If running in a non-interactive environment, plots might not display.")
    print("Uncomment save_figure lines to save plots to files.")

    # Test error case for get_style in Plotter, indirectly
    faulty_config = PlotConfig()
    faulty_config.themes["light"].pop("background_color") # Damage the config
    plotter_faulty = Plotter(config=faulty_config)
    try:
        fig_faulty, ax_faulty = plotter_faulty.create_figure_and_axes()
        print("Faulty plot created (this should not happen if key is missing).")
        plt.close(fig_faulty)
    except KeyError as e:
        print(f"\nSuccessfully caught expected error for missing key: {e}")

    print("Plotter class defined and example usage tested.")
