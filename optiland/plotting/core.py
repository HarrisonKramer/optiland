"""Core plotting functionalities for the Optiland library.

This module provides the main `Plotter` class, which is the primary interface
for creating various types of plots. It integrates with the `config` and
`themes` modules to allow for extensive customization of plot appearance.
"""

import matplotlib.pyplot as plt
import numpy as np  # Added for examples

from . import config, exceptions, themes


# Helper function (previously _apply_theme_and_config_to_ax)
def _apply_theme_and_config_to_ax_static(ax, theme_settings: dict, plot_configs: dict):
    """Applies theme and configuration settings to a matplotlib Axes object.

    This is a helper function to centralize common styling operations.

    Args:
      ax: The matplotlib.axes.Axes object to style.
      theme_settings: A dictionary of theme settings.
      plot_configs: A dictionary of relevant plot configurations.

    """
    # Apply property cycler first
    prop_cycle = theme_settings.get("axes.prop_cycle")
    if prop_cycle:
        ax.set_prop_cycle(prop_cycle)

    ax.set_facecolor(theme_settings.get("axes.facecolor", "#FFFFFF"))
    ax.grid(
        True,  # TODO: Make grid state configurable
        color=theme_settings.get("grid.color", "#D0D0D0"),
        alpha=theme_settings.get("grid.alpha", 0.8),
    )
    ax.spines["top"].set_edgecolor(theme_settings.get("axes.edgecolor", "#333333"))
    ax.spines["bottom"].set_edgecolor(theme_settings.get("axes.edgecolor", "#333333"))
    ax.spines["left"].set_edgecolor(theme_settings.get("axes.edgecolor", "#333333"))
    ax.spines["right"].set_edgecolor(theme_settings.get("axes.edgecolor", "#333333"))

    # Apply font sizes from config
    ax.title.set_fontsize(plot_configs.get("font.size_title", 16))
    ax.xaxis.label.set_fontsize(plot_configs.get("font.size_label", 14))
    ax.yaxis.label.set_fontsize(plot_configs.get("font.size_label", 14))
    ax.tick_params(
        labelsize=plot_configs.get("font.size_legend", 12),
    )  # Using legend size for 2D tick labels

    if hasattr(ax, "zaxis"):  # 3D specific settings
        ax.zaxis.label.set_fontsize(plot_configs.get("font.size_label", 14))
        # Apply pane colors for 3D plots
        pane_color = theme_settings.get(
            "axes3d.pane_color",
            theme_settings.get("axes.facecolor", "#EAEAF2"),
        )
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)

        # Apply 3D grid color
        grid3d_color = theme_settings.get(
            "axes3d.grid_color",
            theme_settings.get("grid.color", "#D0D0D0"),
        )
        ax.w_xaxis._axinfo.update(
            {"grid": {"color": grid3d_color, "linewidth": 0.5, "linestyle": ":"}},
        )
        ax.w_yaxis._axinfo.update(
            {"grid": {"color": grid3d_color, "linewidth": 0.5, "linestyle": ":"}},
        )
        ax.w_zaxis._axinfo.update(
            {"grid": {"color": grid3d_color, "linewidth": 0.5, "linestyle": ":"}},
        )

        # Set face color for the 3D axes background itself
        ax.set_facecolor(
            theme_settings.get(
                "axes3d.facecolor",
                theme_settings.get("axes.facecolor", "#FFFFFF"),
            ),
        )

        # Tick label colors for 3D (already handled by tick_params if general,
        # but can be specific)
        ax.tick_params(axis="x", colors=theme_settings.get("xtick.color", "#333333"))
        ax.tick_params(axis="y", colors=theme_settings.get("ytick.color", "#333333"))
        ax.tick_params(
            axis="z",
            colors=theme_settings.get("xtick.color", "#333333"),
        )  # Assuming ztick uses similar color
    else:  # 2D specific settings that might not apply or need adjustment for 3D
        ax.grid(
            True,  # TODO: Make grid state configurable
            color=theme_settings.get("grid.color", "#D0D0D0"),
            alpha=theme_settings.get("grid.alpha", 0.8),
        )
        ax.spines["top"].set_edgecolor(theme_settings.get("axes.edgecolor", "#333333"))
        ax.spines["bottom"].set_edgecolor(
            theme_settings.get("axes.edgecolor", "#333333"),
        )
        ax.spines["left"].set_edgecolor(theme_settings.get("axes.edgecolor", "#333333"))
        ax.spines["right"].set_edgecolor(
            theme_settings.get("axes.edgecolor", "#333333"),
        )


class Plotter:
    """A class to create various plots using matplotlib with custom themes and configs.

    The `Plotter` class simplifies the process of generating common plots
    by integrating theme and configuration management directly into the
    plotting methods. It allows users to easily switch themes, update
    global plot settings, and create visualizations with consistent styling.

    Example:
        >>> from optiland.plotting import Plotter
        >>> plotter = Plotter()
        >>> x_values = list(range(5))
        >>> y_values = [x*x for x in x_values]
        >>> plotter.plot_line(x_values, y_values, title="My First Plot")
        # This will display the plot based on current theme and config.

        >>> plotter.set_theme("dark")
        >>> plotter.update_config("font.size_title", 20)
        >>> plotter.plot_scatter(x_values, y_values, title="Dark Scatter, Large Title")
        # Plot is shown with dark theme and a larger title font.

    """

    def __init__(self):
        """Initializes the Plotter.

        The plotter relies on the global settings defined in the
        `optiland.plotting.config` and `optiland.plotting.themes` modules.
        """
        # No specific state needed in __init__ for now, as it uses global
        # config/theme modules.

    def set_theme(self, theme_name: str):
        """Sets the active theme for all subsequent plots generated by this plotter.

        Args:
          theme_name: The name of the theme to activate (e.g., 'light', 'dark').

        Raises:
          ValueError: If the theme_name is not a defined theme.

        """
        themes.set_active_theme(theme_name)

    def update_config(self, key: str, value):
        """Updates a global plotting configuration setting.

        This change will affect all subsequent plots.

        Args:
          key: The dot-separated configuration key (e.g., 'figure.figsize').
          value: The new value for the configuration setting.

        Raises:
          KeyError: If the key is not a valid configuration option.
          TypeError: If the value type is incorrect for the given key.

        """
        config.set_config(key, value)

    def get_current_theme_settings(self) -> dict:
        """Retrieves the dictionary of settings for the currently active theme.

        Returns:
          A dictionary containing all key-value pairs for the active theme.

        """
        return themes.get_active_theme_dict()

    def get_config_value(self, key: str):
        """Retrieves a specific configuration value.

        Args:
          key: The dot-separated configuration key.

        Returns:
          The value of the configuration setting.

        Raises:
          KeyError: If the key is not found.

        """
        return config.get_config(key)

    @staticmethod
    def plot_line(
        x,
        y,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        legend_label: str = None,
        return_fig_ax: bool = None,
        ax=None,  # New parameter
        **kwargs,
    ):
        """Plots a line graph (static method).

        Generates a line plot with the given data, applying current theme and
        configuration settings.

        Args:
          x: Array-like data for the x-axis.
          y: Array-like data for the y-axis.
          title: Optional title for the plot.
          xlabel: Optional label for the x-axis.
          ylabel: Optional label for the y-axis.
          legend_label: Optional label for the line, used if a legend is shown.
          return_fig_ax: Optional. If True, returns (fig, ax). If False, displays
              the plot and returns None. If None, uses the global
              'plot.return_fig_ax' and 'plot.show_on_draw' config.
          ax: Optional. A matplotlib.axes.Axes object to plot on. If None,
              a new figure and axes are created.
          **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.plot()`.

        Returns:
          Optionally, a tuple (matplotlib.figure.Figure, matplotlib.axes.Axes)
          if `return_fig_ax` is True or if the global config 'plot.return_fig_ax'
          is True and `return_fig_ax` is None. Otherwise, returns None.

        Raises:
          exceptions.InvalidPlotDataError: If input data is unsuitable.
          exceptions.ConfigurationError: If a configuration key is missing.
          exceptions.ThemeNotFoundError: If a theme key is missing.

        Example:
            >>> from optiland.plotting import Plotter
            >>> x = [i/10.0 for i in range(50)]
            >>> y = [val**2 for val in x]
            >>> Plotter.plot_line(x, y, title="Squared Values",
            ...                   xlabel="X-value", ylabel="Y-value",
            ...                   legend_label="y = x^2")
            # The plot is displayed automatically if 'plot.show_on_draw' is True.

            To get fig and ax objects for further customization:
            >>> fig, ax = Plotter.plot_line(
            ...     x, y, title="Customized Plot", return_fig_ax=True
            ... )
            >>> if fig and ax: # Check if objects were returned
            ...     ax.text(
            ...         0.5, 0.5, "Custom Annotation",
            ...         transform=ax.transAxes, # type: ignore
            ...         ha='center', color='blue', fontsize=12 # type: ignore
            ...     )
            ...     # If 'plot.show_on_draw' is False (globally or due to
            ...     # return_fig_ax=True), and you need to show it:
            ...     # import matplotlib.pyplot as plt # type: ignore
            ...     # plt.show() # type: ignore

        Example:
            >>> from optiland.plotting import Plotter, themes, config
            >>> x_coords = [1, 2.5, 3, 4.2, 5]
            >>> y_coords = [2, 3, 2.5, 4, 3.5]

            First, a simple scatter plot with default settings:
            >>> Plotter.plot_scatter(x_coords, y_coords, title="Default Scatter Plot")

            Now, customize theme and configuration
            (can also use Plotter instance methods):
            >>> themes.set_active_theme('dark')
            >>> config.set_config('lines.markersize', 10) # Affects scatter marker size
            >>> config.set_config('figure.figsize', (8, 8)) # Make figure larger
            >>> fig, ax = Plotter.plot_scatter(
            ...     x_coords, y_coords,
            ...     title="Dark Scatter with Large Markers",
            ...     xlabel="X Coordinate", ylabel="Y Coordinate",
            ...     legend_label="Data Points",
            ...     return_fig_ax=True
            ... )
            >>> if fig and ax:
            ...    # Add custom grid lines
            ...    ax.grid(True, linestyle=':', linewidth='0.5', color='gray')
            ...    # import matplotlib.pyplot as plt # type: ignore
            ...    # plt.show() # If needed # type: ignore

        """
        theme_settings = themes.get_active_theme_dict()
        fig_created_internally = False

        if return_fig_ax is None:
            should_return_objects = config.get_config("plot.return_fig_ax")
            show_plot_on_draw = config.get_config("plot.show_on_draw")
        else:
            should_return_objects = return_fig_ax
            show_plot_on_draw = not return_fig_ax

        if ax is None:
            fig, ax = plt.subplots(figsize=config.get_config("figure.figsize"))
            fig.set_facecolor(theme_settings.get("figure.facecolor", "#FFFFFF"))
            fig_created_internally = True
        else:
            fig = ax.get_figure()
            # Note: We don't re-apply figure facecolor if ax is provided,
            # assuming fig is already styled.

        try:
            # Basic data validation
            if not (
                hasattr(x, "__len__") and hasattr(y, "__len__") and len(x) == len(y)
            ):
                raise exceptions.InvalidPlotDataError(
                    "x and y must be array-like and have the same length.",
                )
            if len(x) == 0:
                raise exceptions.InvalidPlotDataError(
                    "Input data x and y cannot be empty.",
                )

            plot_configs = {
                "font.size_title": config.get_config("font.size_title"),
                "font.size_label": config.get_config("font.size_label"),
                "font.size_legend": config.get_config("font.size_legend"),
            }
            _apply_theme_and_config_to_ax_static(ax, theme_settings, plot_configs)

            # Plot data
            plot_kwargs = {}
            user_provided_color = kwargs.pop("color", None)
            if user_provided_color:
                plot_kwargs["color"] = user_provided_color
            # If color is not in kwargs, it will use the prop_cycle
            # or matplotlib default.

            plot_kwargs["linewidth"] = kwargs.pop(
                "linewidth",
                config.get_config("lines.linewidth"),
            )
            plot_kwargs["markersize"] = kwargs.pop(
                "markersize",
                config.get_config("lines.markersize"),
            )
            plot_kwargs.update(kwargs)  # Add any remaining kwargs

            ax.plot(x, y, label=legend_label, **plot_kwargs)

            # Set title and labels
            if title:
                ax.set_title(
                    title,
                    color=theme_settings.get("axes.titlecolor", "#333333"),
                    fontsize=config.get_config("font.size_title"),
                )
            if xlabel:
                ax.set_xlabel(
                    xlabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=config.get_config("font.size_label"),
                )
            if ylabel:
                ax.set_ylabel(
                    ylabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=config.get_config("font.size_label"),
                )

            # Set tick colors
            ax.tick_params(
                axis="x",
                colors=theme_settings.get("xtick.color", "#333333"),
            )
            ax.tick_params(
                axis="y",
                colors=theme_settings.get("ytick.color", "#333333"),
            )

            # Add legend
            if legend_label:
                legend = ax.legend(fontsize=config.get_config("font.size_legend"))
                # Set legend text color (requires iterating through texts)
                legend_text_color = theme_settings.get(
                    "legend.labelcolor",
                    theme_settings.get("text.color", "#333333"),
                )  # type: ignore
                for text in legend.get_texts():
                    text.set_color(legend_text_color)
                if theme_settings.get("legend.facecolor"):
                    legend.get_frame().set_facecolor(
                        theme_settings.get("legend.facecolor"),
                    )
                if theme_settings.get("legend.edgecolor"):
                    legend.get_frame().set_edgecolor(
                        theme_settings.get("legend.edgecolor"),
                    )

            if show_plot_on_draw and not should_return_objects:
                plt.show()

            if should_return_objects:
                return fig, ax
            plt.close(fig)
            return None
        except exceptions.PlottingError as e:  # Re-raise our own plotting errors
            if (
                fig_created_internally and "fig" in locals()
            ):  # Check if fig was defined before trying to close
                plt.close(fig)
            raise e
        except Exception as e:  # Wrap other unexpected errors
            if fig_created_internally and "fig" in locals():  # Check if fig was defined
                plt.close(fig)
            raise exceptions.PlottingError(
                f"An unexpected error occurred in plot_line: {e}",
            ) from e

    @staticmethod
    def plot_image(
        image_data,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        cmap: str = None,
        show_colorbar: bool = True,
        return_fig_ax: bool = None,
        ax=None,  # New parameter
        **kwargs,
    ):
        """Plots 2D image data, such as a heatmap (static method).

        Generates an image plot using `imshow`, applying current theme and
        configuration settings.

        Args:
          image_data: A 2D array-like structure (e.g., NumPy array) containing
              the image data.
          title: Optional title for the plot.
          xlabel: Optional label for the x-axis.
          ylabel: Optional label for the y-axis.
          cmap: Optional colormap name (string). If None, uses the
              'image.cmap' value from the global configuration.
          show_colorbar: If True, a colorbar is added to the plot.
          return_fig_ax: Optional. If True, returns (fig, ax). If False, displays
              the plot and returns None. If None, uses the global
              'plot.return_fig_ax' and 'plot.show_on_draw' config.
          ax: Optional. A matplotlib.axes.Axes object to plot on. If None,
              a new figure and axes are created.
          **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.imshow()`.

        Returns:
          Optionally, a tuple (matplotlib.figure.Figure, matplotlib.axes.Axes)
          if `return_fig_ax` is True or if the global config 'plot.return_fig_ax'
          is True and `return_fig_ax` is None. Otherwise, returns None.

        Raises:
          exceptions.InvalidPlotDataError: If input data is unsuitable.
          exceptions.ConfigurationError: If a configuration key is missing.
          exceptions.ThemeNotFoundError: If a theme key is missing.

        Example:
            >>> import numpy as np
            >>> from optiland.plotting import Plotter, themes, config
            >>> data = np.random.rand(10, 10)
            >>> Plotter.plot_image(data, title="Heatmap Example",
            ...                    xlabel="X-axis", ylabel="Y-axis",
            ...                    cmap='viridis')
            # Plot is displayed with viridis colormap.

            >>> # Can use plotter_instance.set_theme('dark')
            >>> themes.set_active_theme('dark') # noqa: E501
            >>> fig, ax = Plotter.plot_image(
            ...    data, title="Dark Heatmap with different cmap",
            ...    cmap='inferno', return_fig_ax=True
            ... )
            >>> if fig and ax:
            ...     # Example: remove x-tick labels
            ...     ax.set_xticklabels([]) # type: ignore # noqa: E501
            ...     # import matplotlib.pyplot as plt # type: ignore
            ...     # plt.show() # If needed # type: ignore

        """
        theme_settings = themes.get_active_theme_dict()
        fig_created_internally = False

        if return_fig_ax is None:
            should_return_objects = config.get_config("plot.return_fig_ax")
            show_plot_on_draw = config.get_config("plot.show_on_draw")
        else:
            should_return_objects = return_fig_ax
            show_plot_on_draw = not return_fig_ax

        if ax is None:
            fig, ax = plt.subplots(figsize=config.get_config("figure.figsize"))
            fig.set_facecolor(theme_settings.get("figure.facecolor", "#FFFFFF"))
            fig_created_internally = True
        else:
            fig = ax.get_figure()

        try:
            # Basic data validation for image_data
            img_array = np.asarray(image_data)
            if img_array.ndim != 2:
                raise exceptions.InvalidPlotDataError(
                    f"image_data must be a 2D array-like structure. "
                    f"Got {img_array.ndim} dimensions."
                )
            if img_array.size == 0:
                raise exceptions.InvalidPlotDataError("image_data cannot be empty.")

            plot_configs = {
                "font.size_title": config.get_config("font.size_title"),
                "font.size_label": config.get_config("font.size_label"),
                "font.size_legend": config.get_config(
                    "font.size_legend",
                ),  # Used for tick labels by _apply_theme
            }
            _apply_theme_and_config_to_ax_static(ax, theme_settings, plot_configs)

            current_cmap = cmap or config.get_config("image.cmap")
            im = ax.imshow(image_data, cmap=current_cmap, **kwargs)

            if title:
                ax.set_title(
                    title,
                    color=theme_settings.get("axes.titlecolor", "#333333"),
                    fontsize=config.get_config("font.size_title"),
                )
            if xlabel:
                ax.set_xlabel(
                    xlabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=config.get_config("font.size_label"),
                )
            if ylabel:
                ax.set_ylabel(
                    ylabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=config.get_config("font.size_label"),
                )

            ax.tick_params(
                axis="x",
                colors=theme_settings.get("xtick.color", "#333333"),
            )
            ax.tick_params(
                axis="y",
                colors=theme_settings.get("ytick.color", "#333333"),
            )

            if show_colorbar:
                cb = fig.colorbar(im, ax=ax)
                # Basic theming for colorbar ticks
                cb.ax.tick_params(
                    colors=theme_settings.get("ytick.color", "#333333"),  # type: ignore
                    labelsize=config.get_config("font.size_legend") - 2,
                )  # smaller font for colorbar

            if show_plot_on_draw and not should_return_objects:
                plt.show()

            if should_return_objects:
                return fig, ax
            plt.close(fig)
            return None
        except exceptions.PlottingError as e:
            if fig_created_internally and "fig" in locals():
                plt.close(fig)
            raise e
        except Exception as e:
            if fig_created_internally and "fig" in locals():
                plt.close(fig)
            raise exceptions.PlottingError(
                f"An unexpected error occurred in plot_image: {e}",
            ) from e

    @staticmethod
    def plot_subplots(
        num_rows: int,
        num_cols: int,
        plot_callbacks: list,
        sharex: bool = False,
        sharey: bool = False,
        main_title: str = None,
        return_fig_axs: bool = None,
        **fig_kwargs,
    ):
        """Creates a figure with multiple subplots arranged in a grid (static method).

        Each subplot is populated by a user-provided callback function.

        Args:
          num_rows: Number of rows in the subplot grid.
          num_cols: Number of columns in the subplot grid.
          plot_callbacks: A list of functions. Each function must accept two
              arguments: a `matplotlib.axes.Axes` object (the subplot's axes)
              and an integer `index` (the 0-based index of the subplot in
              the flattened grid). The callback is responsible for plotting data
              on the provided `ax`.
          sharex: Whether subplots should share the x-axis. See `plt.subplots`.
          sharey: Whether subplots should share the y-axis. See `plt.subplots`.
          main_title: Optional super-title for the entire figure.
          return_fig_axs: Optional. If True, returns (fig, axs). If False, displays
              the plot and returns None. If None, uses the global
              'plot.return_fig_ax' (note: singular 'ax' in config name)
              and 'plot.show_on_draw' config.
          **fig_kwargs: Additional keyword arguments passed to
                        `matplotlib.pyplot.subplots()`.

        Returns:
          Optionally, a tuple (matplotlib.figure.Figure, array of matplotlib.axes.Axes)
          if `return_fig_axs` is True or if the global config 'plot.return_fig_ax'
          is True and `return_fig_axs` is None. Otherwise, returns None.

        Raises:
          ValueError: If the number of callbacks does not match `num_rows * num_cols`
                      (this is a specific argument validation, kept as ValueError).
          exceptions.ConfigurationError: If a configuration key is missing.
          exceptions.ThemeNotFoundError: If a theme key is missing.
          exceptions.InvalidPlotDataError: If plot_callbacks is not a list or is empty.


        Example:
            >>> import numpy as np
            >>> from optiland.plotting import Plotter
            >>>
            >>> def plot_sine(ax, index):
            ...     x = np.linspace(0, 2 * np.pi, 100)
            ...     y = np.sin(x + index * np.pi / 2)
            ...     ax.plot(x, y) # Line color will be from theme by default
            ...     ax.set_title( # Title color from theme
            ...          f'Plot {index+1}: Sin(x + {index}\\u03C0/2)'
            ...     )
            ...     # ax.set_xlabel("X") # Labels can be set here too
            ...     # ax.set_ylabel("Y")
            >>>
            >>> def plot_random_scatter(ax, index):
            ...     x_scatter = np.random.rand(20)
            ...     y_scatter = np.random.rand(20)
            ...     # Example of overriding theme color for this specific subplot
            ...     ax.scatter(x_scatter, y_scatter, color='purple', s=50)
            ...     ax.set_title(f'Plot {index+1}: Random Scatter')
            >>>
            >>> callbacks = [
            ...     plot_sine, plot_random_scatter, plot_sine, plot_random_scatter
            ... ]
            >>> Plotter.plot_subplots(2, 2, callbacks, main_title="Multiple Plots Demo")
            # Plot is displayed.
            >>>
            >>> # Example with returning figure and axes array
            >>> fig, axs = Plotter.plot_subplots(
            ...     1, 2, [plot_sine, plot_random_scatter],
            ...     main_title="Two Subplots", return_fig_axs=True
            ... )
            >>> # Check if objects were returned (axs is a numpy array)
            >>> if fig and axs.size == 2: # noqa: E501
            ...    axs[0].set_xlabel("X-axis for Sine") # type: ignore
            ...    axs[1].set_ylabel("Y-axis for Scatter") # type: ignore
            ...    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for main_title
            ...    # import matplotlib.pyplot as plt # type: ignore
            ...    # plt.show() # If needed # type: ignore

        """
        # Initial argument validation
        if not isinstance(plot_callbacks, list) or not plot_callbacks:
            raise exceptions.InvalidPlotDataError(
                "plot_callbacks must be a non-empty list of functions.",
            )
        if len(plot_callbacks) != num_rows * num_cols:
            raise ValueError(  # Keeping ValueError for direct argument validation
                f"Number of plot_callbacks ({len(plot_callbacks)}) must match "
                f"num_rows * num_cols ({num_rows * num_cols}).",
            )
        if not all(callable(cb) for cb in plot_callbacks):
            raise exceptions.InvalidPlotDataError(
                "All items in plot_callbacks must be callable functions.",
            )

        theme_settings = themes.get_active_theme_dict()

        # Determine return/show behavior
        if return_fig_axs is None:
            should_return_objects = config.get_config("plot.return_fig_ax")
            show_plot_on_draw = config.get_config("plot.show_on_draw")
        else:
            should_return_objects = return_fig_axs
            show_plot_on_draw = not return_fig_axs

        fig = None  # Initialize fig to None for cleanup in case of early error
        try:
            fig, axs = plt.subplots(
                num_rows,
                num_cols,
                sharex=sharex,
                sharey=sharey,
                figsize=config.get_config("figure.figsize"),
                **fig_kwargs,
            )
            fig.set_facecolor(theme_settings.get("figure.facecolor", "#FFFFFF"))

            plot_configs = {
                "font.size_title": config.get_config("font.size_title"),
                "font.size_label": config.get_config("font.size_label"),
                "font.size_legend": config.get_config("font.size_legend"),
            }

            if main_title:
                fig.suptitle(
                    main_title,
                    fontsize=plot_configs["font.size_title"]
                    + 2,  # Slightly larger for main title
                    color=theme_settings.get("axes.titlecolor", "#333333"),
                    # y=0.98 # Adjust position if needed
                )

            # Ensure axs is always an array, even for single subplot case (e.g. 1,1)
            axs_flat = [axs] if not isinstance(axs, np.ndarray) else axs.flat

            for i, ax_subplot in enumerate(axs_flat):
                if i < len(plot_callbacks):
                    _apply_theme_and_config_to_ax_static(
                        ax_subplot,
                        theme_settings,
                        plot_configs,
                    )
                    # Titles and labels within callbacks will also use theme colors
                    # if they use ax.set_title, ax.set_xlabel etc. directly.
                    plot_callbacks[i](ax_subplot, i)

            # Improve layout, especially if there's a suptitle
            if main_title:
                fig.tight_layout(
                    rect=[0, 0.03, 1, 0.95],
                )  # Adjust rect to make space for suptitle
            else:
                fig.tight_layout()

            if show_plot_on_draw and not should_return_objects:
                plt.show()

            if should_return_objects:
                return fig, axs  # Return the potentially multi-dimensional array `axs`

            # If not returning objects, close the figure explicitly
            plt.close(fig)
            return None

        except exceptions.PlottingError as e:
            if fig is not None:
                plt.close(fig)
            raise e
        except ValueError as e:
            if fig is not None:
                plt.close(fig)
            raise e
        except Exception as e:
            if fig is not None:
                plt.close(fig)
            raise exceptions.PlottingError(
                f"An unexpected error occurred in plot_subplots: {e}",
            ) from e

    @staticmethod
    def plot_line_3d(
        x,
        y,
        z,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        zlabel: str = None,
        legend_label: str = None,
        return_fig_ax: bool = None,
        ax=None,  # New parameter
        **kwargs,
    ):
        """Plots a 3D line graph (static method).

        Generates a 3D line plot with the given data, applying current theme and
        configuration settings.

        Args:
          x: Array-like data for the x-axis.
          y: Array-like data for the y-axis.
          z: Array-like data for the z-axis.
          title: Optional title for the plot.
          xlabel: Optional label for the x-axis.
          ylabel: Optional label for the y-axis.
          zlabel: Optional label for the z-axis.
          legend_label: Optional label for the line, used if a legend is shown.
          return_fig_ax: Optional. If True, returns (fig, ax). If False, displays
              the plot. If None, uses global 'plot.return_fig_ax' and
              'plot.show_on_draw' config.
          ax: Optional. A matplotlib.axes.Axes3D object to plot on. If None,
              a new figure and 3D axes are created.
          **kwargs: Additional keyword arguments passed to `ax.plot()`.

        Returns:
          Optionally, a tuple (matplotlib.figure.Figure, matplotlib.axes.Axes3D)
          if `return_fig_ax` is True or if global config 'plot.return_fig_ax'
          is True and `return_fig_ax` is None. Otherwise, returns None.

        Raises:
          exceptions.InvalidPlotDataError: If input data is unsuitable.
          exceptions.ConfigurationError: If a configuration key is missing.
          exceptions.ThemeNotFoundError: If a theme key is missing.

        Example:
            >>> import numpy as np
            >>> from optiland.plotting import Plotter, themes
            >>> t = np.linspace(-4 * np.pi, 4 * np.pi, 100)
            >>> x = np.sin(t)
            >>> y = np.cos(t)
            >>> z = t / (4 * np.pi)
            >>> Plotter.plot_line_3d(x, y, z, title="3D Helix", legend_label="Path")
            >>> # Can use plotter_instance.set_theme('dark')
            >>> themes.set_active_theme('dark') # noqa: E501
            >>> fig, ax = Plotter.plot_line_3d(
            ...     x, y, z, title="Dark Helix", return_fig_ax=True
            ... )
            >>> if fig and ax:
            ...     ax.view_init(elev=30, azim=60) # type: ignore # Change view angle
            ...     # import matplotlib.pyplot as plt # type: ignore
            ...     # plt.show() # If needed # type: ignore

        """
        theme_settings = themes.get_active_theme_dict()
        fig_created_internally = False

        if return_fig_ax is None:
            should_return_objects = config.get_config("plot.return_fig_ax")
            show_plot_on_draw = config.get_config("plot.show_on_draw")
        else:
            should_return_objects = return_fig_ax
            show_plot_on_draw = not return_fig_ax

        if ax is None:
            fig = plt.figure(figsize=config.get_config("figure.figsize"))
            fig.set_facecolor(theme_settings.get("figure.facecolor", "#FFFFFF"))
            ax = fig.add_subplot(111, projection="3d")
            fig_created_internally = True
        else:
            fig = ax.get_figure()
            if not hasattr(ax, "zaxis"):
                raise ValueError("Provided 'ax' for 3D plot must be a 3D Axes object.")

        try:
            if not (
                hasattr(x, "__len__")
                and hasattr(y, "__len__")
                and hasattr(z, "__len__")
                and len(x) == len(y) == len(z)
            ):
                raise exceptions.InvalidPlotDataError(
                    "x, y, and z must be array-like and have the same length.",
                )
            if len(x) == 0:
                raise exceptions.InvalidPlotDataError(
                    "Input data x, y, z cannot be empty.",
                )

            plot_configs = {
                "font.size_title": config.get_config("font.size_title"),
                "font.size_label": config.get_config("font.size_label"),
                "font.size_legend": config.get_config("font.size_legend"),
            }
            # This will now also handle 3D specific theming like pane colors
            _apply_theme_and_config_to_ax_static(ax, theme_settings, plot_configs)

            plot_kwargs = {}
            user_provided_color = kwargs.pop("color", None)
            if user_provided_color:
                plot_kwargs["color"] = user_provided_color

            plot_kwargs["linewidth"] = kwargs.pop(
                "linewidth",
                config.get_config("lines.linewidth"),
            )
            plot_kwargs.update(kwargs)

            ax.plot(x, y, z, label=legend_label, **plot_kwargs)

            if title:
                ax.set_title(
                    title,
                    color=theme_settings.get("axes.titlecolor", "#333333"),
                    fontsize=plot_configs["font.size_title"],
                )
            if xlabel:
                ax.set_xlabel(
                    xlabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=plot_configs["font.size_label"],
                )
            if ylabel:
                ax.set_ylabel(
                    ylabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=plot_configs["font.size_label"],
                )
            if zlabel:
                ax.set_zlabel(
                    zlabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=plot_configs["font.size_label"],
                )

            # Tick colors are handled by _apply_theme_and_config_to_ax_static for 3D

            if legend_label:
                legend = ax.legend(fontsize=plot_configs["font.size_legend"])
                legend_text_color = theme_settings.get(
                    "legend.labelcolor",
                    theme_settings.get("text.color", "#333333"),
                )  # type: ignore
                for text_obj in (
                    legend.get_texts()
                ):  # Matplotlib uses 'text' not 'labelcolor' for legend
                    text_obj.set_color(legend_text_color)  # type: ignore
                if theme_settings.get("legend.facecolor"):
                    legend.get_frame().set_facecolor(
                        theme_settings.get("legend.facecolor"),
                    )
                if theme_settings.get("legend.edgecolor"):
                    legend.get_frame().set_edgecolor(
                        theme_settings.get("legend.edgecolor"),
                    )

            if show_plot_on_draw and not should_return_objects:
                plt.show()

            if should_return_objects:
                return fig, ax
            plt.close(fig)
            return None
        except exceptions.PlottingError as e:
            if (
                fig_created_internally and "fig" in locals()
            ):  # Check if fig was defined before trying to close
                plt.close(fig)
            raise e
        except Exception as e:
            if fig_created_internally and "fig" in locals():  # Check if fig was defined
                plt.close(fig)
            raise exceptions.PlottingError(
                f"An unexpected error occurred in plot_line_3d: {e}",
            ) from e

    @staticmethod
    def plot_scatter_3d(
        x,
        y,
        z,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        zlabel: str = None,
        # Note: legend for scatter3d can be tricky if colors/sizes vary per point
        legend_label: str = None,
        return_fig_ax: bool = None,
        ax=None,  # New parameter
        **kwargs,
    ):
        """Plots a 3D scatter graph (static method).

        Generates a 3D scatter plot, applying current theme and configuration.

        Args:
          x: Array-like data for the x-axis.
          y: Array-like data for the y-axis.
          z: Array-like data for the z-axis.
          title: Optional title for the plot.
          xlabel: Optional label for the x-axis.
          ylabel: Optional label for the y-axis.
          zlabel: Optional label for the z-axis.
          legend_label: Optional label for the points, used if a legend is shown.
          return_fig_ax: Optional. If True, returns (fig, ax). If False, displays.
              If None, uses global config.
          ax: Optional. A matplotlib.axes.Axes3D object to plot on. If None,
              a new figure and 3D axes are created.
          **kwargs: Additional keyword arguments passed to `ax.scatter()`.
              Useful for `c` (color array), `s` (size array), `marker`.

        Returns:
          Optionally, a tuple (matplotlib.figure.Figure, matplotlib.axes.Axes3D)
          if `return_fig_ax` is True or if global config 'plot.return_fig_ax'
          is True and `return_fig_ax` is None. Otherwise, returns None.

        Raises:
          exceptions.InvalidPlotDataError: If input data is unsuitable.
          exceptions.ConfigurationError: If a configuration key is missing.
          exceptions.ThemeNotFoundError: If a theme key is missing.

        Example:
            >>> import numpy as np
            >>> from optiland.plotting import Plotter
            >>> x_rand = np.random.rand(50)
            >>> y_rand = np.random.rand(50)
            >>> z_rand = np.random.rand(50)
            >>> colors = np.random.rand(50) # Per-point color
            >>> sizes = 100 * np.random.rand(50) # Per-point size
            >>> Plotter.plot_scatter_3d(
            ...     x_rand, y_rand, z_rand, title="3D Scatter Points",
            ...     c=colors, s=sizes, cmap='viridis', # noqa: E501
            ...     legend_label="Random Data"
            ... )
            # Note: legend_label might not be effective if 'c' is an array for colors.
            # A colorbar might be more appropriate if 'c' maps to a cmap.

        """
        theme_settings = themes.get_active_theme_dict()
        fig_created_internally = False

        if return_fig_ax is None:
            should_return_objects = config.get_config("plot.return_fig_ax")
            show_plot_on_draw = config.get_config("plot.show_on_draw")
        else:
            should_return_objects = return_fig_ax
            show_plot_on_draw = not return_fig_ax

        if ax is None:
            fig = plt.figure(figsize=config.get_config("figure.figsize"))
            fig.set_facecolor(theme_settings.get("figure.facecolor", "#FFFFFF"))
            ax = fig.add_subplot(111, projection="3d")
            fig_created_internally = True
        else:
            fig = ax.get_figure()
            if not hasattr(ax, "zaxis"):
                raise ValueError("Provided 'ax' for 3D plot must be a 3D Axes object.")

        try:
            if not (
                hasattr(x, "__len__")
                and hasattr(y, "__len__")
                and hasattr(z, "__len__")
                and len(x) == len(y) == len(z)
            ):
                raise exceptions.InvalidPlotDataError(
                    "x, y, and z must be array-like and have the same length.",
                )
            if len(x) == 0:
                raise exceptions.InvalidPlotDataError(
                    "Input data x, y, z cannot be empty.",
                )

            plot_configs = {
                "font.size_title": config.get_config("font.size_title"),
                "font.size_label": config.get_config("font.size_label"),
                "font.size_legend": config.get_config("font.size_legend"),
            }
            _apply_theme_and_config_to_ax_static(ax, theme_settings, plot_configs)

            # Scatter plot specific settings
            plot_kwargs = {}
            user_provided_color = kwargs.pop(
                "color",
                kwargs.pop("c", None),
            )  # 'c' is an alias for color
            if user_provided_color:
                plot_kwargs["color"] = user_provided_color

            # Default marker size from config (squared for scatter 's'),
            # allow override by 's'
            plot_kwargs["s"] = kwargs.pop(
                "s",
                config.get_config("lines.markersize") ** 2,
            )
            plot_kwargs.update(kwargs)

            ax.scatter(x, y, z, label=legend_label, **plot_kwargs)

            if title:
                ax.set_title(
                    title,
                    color=theme_settings.get("axes.titlecolor", "#333333"),
                    fontsize=plot_configs["font.size_title"],
                )
            if xlabel:
                ax.set_xlabel(
                    xlabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=plot_configs["font.size_label"],
                )
            if ylabel:
                ax.set_ylabel(
                    ylabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=plot_configs["font.size_label"],
                )
            if zlabel:
                ax.set_zlabel(
                    zlabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=plot_configs["font.size_label"],
                )

            if (
                legend_label
            ):  # Legend for 3D scatter can be simple if color/size is uniform
                legend = ax.legend(fontsize=plot_configs["font.size_legend"])
                legend_text_color = theme_settings.get(
                    "legend.labelcolor",
                    theme_settings.get("text.color", "#333333"),
                )  # type: ignore
                for text_obj in legend.get_texts():
                    text_obj.set_color(legend_text_color)  # type: ignore
                if theme_settings.get("legend.facecolor"):
                    legend.get_frame().set_facecolor(
                        theme_settings.get("legend.facecolor"),
                    )
                if theme_settings.get("legend.edgecolor"):
                    legend.get_frame().set_edgecolor(
                        theme_settings.get("legend.edgecolor"),
                    )

            if show_plot_on_draw and not should_return_objects:
                plt.show()

            if should_return_objects:
                return fig, ax
            plt.close(fig)
            return None
        except exceptions.PlottingError as e:
            if fig_created_internally and "fig" in locals():
                plt.close(fig)
            raise e
        except Exception as e:
            if fig_created_internally and "fig" in locals():
                plt.close(fig)
            raise exceptions.PlottingError(
                f"An unexpected error occurred in plot_scatter_3d: {e}",
            ) from e

    @staticmethod
    def plot_surface(
        X,
        Y,
        Z,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        zlabel: str = None,
        cmap: str = None,
        show_colorbar: bool = True,
        return_fig_ax: bool = None,
        ax=None,  # New parameter
        **kwargs,
    ):
        """Plots a 3D surface (static method).

        Generates a 3D surface plot, applying theme and configuration.

        Args:
          X: 2D array of x-coordinates (e.g., from np.meshgrid).
          Y: 2D array of y-coordinates (e.g., from np.meshgrid).
          Z: 2D array of z-values corresponding to (X, Y).
          title: Optional title for the plot.
          xlabel: Optional label for the x-axis.
          ylabel: Optional label for the y-axis.
          zlabel: Optional label for the z-axis.
          cmap: Optional colormap name. Defaults to 'image.cmap' from config.
          show_colorbar: If True, adds a colorbar.
          return_fig_ax: Optional. If True, returns (fig, ax). If False, displays.
              If None, uses global config.
          ax: Optional. A matplotlib.axes.Axes3D object to plot on. If None,
              a new figure and 3D axes are created.
          **kwargs: Additional keyword arguments passed to `ax.plot_surface()`.
              Common arguments: `rstride`, `cstride`, `linewidth`, `edgecolor`.

        Returns:
          Optionally, a tuple (matplotlib.figure.Figure, matplotlib.axes.Axes3D)
          if `return_fig_ax` is True or if global config 'plot.return_fig_ax'
          is True and `return_fig_ax` is None. Otherwise, returns None.

        Raises:
          exceptions.InvalidPlotDataError: If input data X, Y, Z are unsuitable
                                           (e.g., not 2D or mismatched shapes).
          exceptions.ConfigurationError: If a configuration key is missing.
          exceptions.ThemeNotFoundError: If a theme key is missing.

        Example:
            >>> import numpy as np
            >>> from optiland.plotting import Plotter
            >>> x_surf = np.arange(-5, 5, 0.25)
            >>> y_surf = np.arange(-5, 5, 0.25)
            >>> X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
            >>> R = np.sqrt(X_surf**2 + Y_surf**2)
            >>> Z_surf = np.sin(R)
            >>> Plotter.plot_surface(
            ...     X_surf, Y_surf, Z_surf, title="3D Surface (Sine Wave)",
            ...     cmap='coolwarm', edgecolor='k', linewidth=0.2  # noqa: E501
            ... )

        """
        theme_settings = themes.get_active_theme_dict()
        fig_created_internally = False

        if return_fig_ax is None:
            should_return_objects = config.get_config("plot.return_fig_ax")
            show_plot_on_draw = config.get_config("plot.show_on_draw")
        else:
            should_return_objects = return_fig_ax
            show_plot_on_draw = not return_fig_ax

        if ax is None:
            fig = plt.figure(figsize=config.get_config("figure.figsize"))
            fig.set_facecolor(theme_settings.get("figure.facecolor", "#FFFFFF"))
            ax = fig.add_subplot(111, projection="3d")
            fig_created_internally = True
        else:
            fig = ax.get_figure()
            if not hasattr(ax, "zaxis"):
                raise ValueError("Provided 'ax' for 3D plot must be a 3D Axes object.")

        try:
            X_arr, Y_arr, Z_arr = np.asarray(X), np.asarray(Y), np.asarray(Z)
            if not (X_arr.ndim == 2 and Y_arr.ndim == 2 and Z_arr.ndim == 2):
                raise exceptions.InvalidPlotDataError(
                    "X, Y, and Z must be 2D array-like structures.",
                )
            if not (X_arr.shape == Y_arr.shape == Z_arr.shape):
                raise exceptions.InvalidPlotDataError(
                    "X, Y, and Z must have the same shape.",
                )
            if X_arr.size == 0:
                raise exceptions.InvalidPlotDataError(
                    "Input data X, Y, Z cannot be empty.",
                )

            plot_configs = {
                "font.size_title": config.get_config("font.size_title"),
                "font.size_label": config.get_config("font.size_label"),
                "font.size_legend": config.get_config("font.size_legend"),
            }
            _apply_theme_and_config_to_ax_static(ax, theme_settings, plot_configs)

            current_cmap = cmap or config.get_config("image.cmap")
            # Default edgecolor from theme if not provided in kwargs
            kwargs.setdefault("edgecolor", theme_settings.get("axes.edgecolor", "none"))
            kwargs.setdefault("linewidth", 0.5)

            surf = ax.plot_surface(X, Y, Z, cmap=current_cmap, **kwargs)

            if title:
                ax.set_title(
                    title,
                    color=theme_settings.get("axes.titlecolor", "#333333"),
                    fontsize=plot_configs["font.size_title"],
                )
            if xlabel:
                ax.set_xlabel(
                    xlabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=plot_configs["font.size_label"],
                )
            if ylabel:
                ax.set_ylabel(
                    ylabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=plot_configs["font.size_label"],
                )
            if zlabel:
                ax.set_zlabel(
                    zlabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=plot_configs["font.size_label"],
                )

            if show_colorbar:
                cb = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
                cb.ax.tick_params(
                    colors=theme_settings.get("ytick.color", "#333333"),  # type: ignore
                    labelsize=plot_configs["font.size_legend"] - 2,
                )

            if show_plot_on_draw and not should_return_objects:
                plt.show()

            if should_return_objects:
                return fig, ax
            plt.close(fig)
            return None
        except exceptions.PlottingError as e:
            if fig_created_internally and "fig" in locals():
                plt.close(fig)
            raise e
        except Exception as e:
            if fig_created_internally and "fig" in locals():
                plt.close(fig)
            raise exceptions.PlottingError(
                f"An unexpected error occurred in plot_surface: {e}",
            ) from e

    @staticmethod
    def plot_wireframe(
        X,
        Y,
        Z,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        zlabel: str = None,
        return_fig_ax: bool = None,
        ax=None,  # New parameter
        **kwargs,
    ):
        """Plots a 3D wireframe (static method).

        Generates a 3D wireframe plot, applying theme and configuration.

        Args:
          X: 2D array of x-coordinates (e.g., from np.meshgrid).
          Y: 2D array of y-coordinates (e.g., from np.meshgrid).
          Z: 2D array of z-values corresponding to (X, Y).
          title: Optional title for the plot.
          xlabel: Optional label for the x-axis.
          ylabel: Optional label for the y-axis.
          zlabel: Optional label for the z-axis.
          return_fig_ax: Optional. If True, returns (fig, ax). If False, displays.
              If None, uses global config.
          ax: Optional. A matplotlib.axes.Axes3D object to plot on. If None,
              a new figure and 3D axes are created.
          **kwargs: Additional keyword arguments passed to `ax.plot_wireframe()`.
              Useful for `rstride`, `cstride`, `color`, `linewidth`.

        Returns:
          Optionally, a tuple (matplotlib.figure.Figure, matplotlib.axes.Axes3D)
          if `return_fig_ax` is True or if global config 'plot.return_fig_ax'
          is True and `return_fig_ax` is None. Otherwise, returns None.

        Raises:
          exceptions.InvalidPlotDataError: If input data X, Y, Z are unsuitable.
          exceptions.ConfigurationError: If a configuration key is missing.
          exceptions.ThemeNotFoundError: If a theme key is missing.

        Example:
            >>> import numpy as np
            >>> from optiland.plotting import Plotter, themes
            >>> x_wire = np.arange(-5, 5, 0.5) # Coarser steps for wireframe
            >>> y_wire = np.arange(-5, 5, 0.5)
            >>> X_wire, Y_wire = np.meshgrid(x_wire, y_wire)
            >>> R_wire = np.sqrt(X_wire**2 + Y_wire**2)
            >>> Z_wire = np.sin(R_wire)
            >>> Plotter.plot_wireframe(
            ...    X_wire, Y_wire, Z_wire,
            ...    title="3D Wireframe (Sine Wave)", linewidth=0.8 # noqa: E501
            ... )
            >>> # Can use plotter_instance.set_theme('dark')
            >>> themes.set_active_theme('dark') # noqa: E501
            >>> fig, ax = Plotter.plot_wireframe(
            ...     X_wire, Y_wire, Z_wire, title="Dark Wireframe", color='cyan',
            ...     return_fig_ax=True # Override theme line color # noqa: E501
            ... )

        """
        theme_settings = themes.get_active_theme_dict()
        fig_created_internally = False

        if return_fig_ax is None:
            should_return_objects = config.get_config("plot.return_fig_ax")
            show_plot_on_draw = config.get_config("plot.show_on_draw")
        else:
            should_return_objects = return_fig_ax
            show_plot_on_draw = not return_fig_ax

        if ax is None:
            fig = plt.figure(figsize=config.get_config("figure.figsize"))
            fig.set_facecolor(theme_settings.get("figure.facecolor", "#FFFFFF"))
            ax = fig.add_subplot(111, projection="3d")
            fig_created_internally = True
        else:
            fig = ax.get_figure()
            if not hasattr(ax, "zaxis"):
                raise ValueError("Provided 'ax' for 3D plot must be a 3D Axes object.")

        try:
            X_arr, Y_arr, Z_arr = np.asarray(X), np.asarray(Y), np.asarray(Z)
            if not (X_arr.ndim == 2 and Y_arr.ndim == 2 and Z_arr.ndim == 2):
                raise exceptions.InvalidPlotDataError(
                    "X, Y, and Z must be 2D array-like structures.",
                )
            if not (X_arr.shape == Y_arr.shape == Z_arr.shape):
                raise exceptions.InvalidPlotDataError(
                    "X, Y, and Z must have the same shape.",
                )
            if X_arr.size == 0:
                raise exceptions.InvalidPlotDataError(
                    "Input data X, Y, Z cannot be empty.",
                )

            plot_configs = {
                "font.size_title": config.get_config("font.size_title"),
                "font.size_label": config.get_config("font.size_label"),
                "font.size_legend": config.get_config("font.size_legend"),
            }
            _apply_theme_and_config_to_ax_static(ax, theme_settings, plot_configs)

            plot_kwargs = {}
            user_provided_color = kwargs.pop("color", None)
            if user_provided_color:
                plot_kwargs["color"] = user_provided_color
            # If color is not in kwargs, it will use the prop_cycle or
            # matplotlib default for wireframes.
            # Matplotlib's plot_wireframe default color is often 'C0',
            # so it works with cyclers.

            plot_kwargs["linewidth"] = kwargs.pop(
                "linewidth",
                config.get_config("lines.linewidth") * 0.75,
            )  # Thinner for wireframes
            plot_kwargs.update(kwargs)

            ax.plot_wireframe(X, Y, Z, **plot_kwargs)

            if title:
                ax.set_title(
                    title,
                    color=theme_settings.get("axes.titlecolor", "#333333"),
                    fontsize=plot_configs["font.size_title"],
                )
            if xlabel:
                ax.set_xlabel(
                    xlabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=plot_configs["font.size_label"],
                )
            if ylabel:
                ax.set_ylabel(
                    ylabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=plot_configs["font.size_label"],
                )
            if zlabel:
                ax.set_zlabel(
                    zlabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=plot_configs["font.size_label"],
                )

            if show_plot_on_draw and not should_return_objects:
                plt.show()

            if should_return_objects:
                return fig, ax
            plt.close(fig)
            return None
        except exceptions.PlottingError as e:
            if fig_created_internally and "fig" in locals():
                plt.close(fig)
            raise e
        except Exception as e:
            if fig_created_internally and "fig" in locals():
                plt.close(fig)
            raise exceptions.PlottingError(
                f"An unexpected error occurred in plot_wireframe: {e}",
            ) from e

    @staticmethod
    def plot_scatter(
        x,
        y,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        # For scatter, legend is more complex if colors/sizes vary
        legend_label: str = None,
        return_fig_ax: bool = None,
        ax=None,  # New parameter
        **kwargs,
    ):
        """Plots a scatter graph (static method).

        Generates a scatter plot with the given data, applying current theme and
        configuration settings.

        Args:
          x: Array-like data for the x-axis.
          y: Array-like data for the y-axis.
          title: Optional title for the plot.
          xlabel: Optional label for the x-axis.
          ylabel: Optional label for the y-axis.
          legend_label: Optional label for the scatter points, used if a legend
                        is shown.
                        Note: For scatter plots with varying point colors/sizes,
                        legend handling can be more complex. This basic version assumes
                        a single legend entry for the main data series.
          return_fig_ax: Optional. If True, returns (fig, ax). If False, displays
              the plot and returns None. If None, uses the global
              'plot.return_fig_ax' and 'plot.show_on_draw' config.
          ax: Optional. A matplotlib.axes.Axes object to plot on. If None,
              a new figure and axes are created.
          **kwargs: Additional keyword arguments passed to
                    `matplotlib.pyplot.scatter()`.
                    Common arguments include `s` for size and `c` or `color` for color.

        Returns:
          Optionally, a tuple (matplotlib.figure.Figure, matplotlib.axes.Axes)
          if `return_fig_ax` is True or if the global config 'plot.return_fig_ax'
          is True and `return_fig_ax` is None. Otherwise, returns None.

        Raises:
          exceptions.InvalidPlotDataError: If input data is unsuitable.
          exceptions.ConfigurationError: If a configuration key is missing.
          exceptions.ThemeNotFoundError: If a theme key is missing.

        """
        theme_settings = themes.get_active_theme_dict()
        fig_created_internally = False

        if return_fig_ax is None:
            should_return_objects = config.get_config("plot.return_fig_ax")
            show_plot_on_draw = config.get_config("plot.show_on_draw")
        else:
            should_return_objects = return_fig_ax
            show_plot_on_draw = not return_fig_ax

        if ax is None:
            fig, ax = plt.subplots(figsize=config.get_config("figure.figsize"))
            fig.set_facecolor(theme_settings.get("figure.facecolor", "#FFFFFF"))
            fig_created_internally = True
        else:
            fig = ax.get_figure()

        try:
            if not (
                hasattr(x, "__len__") and hasattr(y, "__len__") and len(x) == len(y)
            ):
                raise exceptions.InvalidPlotDataError(
                    "x and y must be array-like and have the same length.",
                )
            if len(x) == 0:
                raise exceptions.InvalidPlotDataError(
                    "Input data x and y cannot be empty.",
                )

            plot_configs = {
                "font.size_title": config.get_config("font.size_title"),
                "font.size_label": config.get_config("font.size_label"),
                "font.size_legend": config.get_config("font.size_legend"),
            }
            _apply_theme_and_config_to_ax_static(ax, theme_settings, plot_configs)

            # Scatter plot specific settings
            plot_kwargs = {}
            user_provided_color = kwargs.pop(
                "color",
                kwargs.pop("c", None),
            )  # 'c' is an alias for color
            if user_provided_color:
                plot_kwargs["color"] = user_provided_color
            # If color is not in kwargs, it will use the prop_cycle
            # or matplotlib default.

            plot_kwargs["s"] = kwargs.pop(
                "s",
                config.get_config("lines.markersize") ** 2,
            )  # markersize in config is often for line markers, scatter 's' is area.
            plot_kwargs.update(kwargs)

            ax.scatter(x, y, label=legend_label, **plot_kwargs)

            if title:
                ax.set_title(
                    title,
                    color=theme_settings.get("axes.titlecolor", "#333333"),
                    fontsize=config.get_config("font.size_title"),
                )
            if xlabel:
                ax.set_xlabel(
                    xlabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=config.get_config("font.size_label"),
                )
            if ylabel:
                ax.set_ylabel(
                    ylabel,
                    color=theme_settings.get("axes.labelcolor", "#333333"),
                    fontsize=config.get_config("font.size_label"),
                )

            ax.tick_params(
                axis="x",
                colors=theme_settings.get("xtick.color", "#333333"),
            )
            ax.tick_params(
                axis="y",
                colors=theme_settings.get("ytick.color", "#333333"),
            )

            if legend_label:
                legend = ax.legend(fontsize=config.get_config("font.size_legend"))
                legend_text_color = theme_settings.get(
                    "legend.labelcolor",
                    theme_settings.get("text.color", "#333333"),
                )  # type: ignore
                for text in legend.get_texts():
                    text.set_color(legend_text_color)  # type: ignore
                if theme_settings.get("legend.facecolor"):
                    legend.get_frame().set_facecolor(
                        theme_settings.get("legend.facecolor"),
                    )
                if theme_settings.get("legend.edgecolor"):
                    legend.get_frame().set_edgecolor(
                        theme_settings.get("legend.edgecolor"),
                    )

            if show_plot_on_draw and not should_return_objects:
                plt.show()

            if should_return_objects:
                return fig, ax
            plt.close(fig)
            return None
        except exceptions.PlottingError as e:
            if fig_created_internally and "fig" in locals():
                plt.close(fig)
            raise e
        except Exception as e:
            if fig_created_internally and "fig" in locals():
                plt.close(fig)
            raise exceptions.PlottingError(
                f"An unexpected error occurred in plot_scatter: {e}",
            ) from e
