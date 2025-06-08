"""OptiLand Plotting Module.

This module provides a centralized interface for creating various types of
scientific plots within the OptiLand application. It abstracts away direct
Matplotlib usage, offers theme management (e.g., 'light', 'dark'), and
allows for global plot configuration.

The main goal is to simplify the creation of consistent and aesthetically
pleasing visualizations with minimal boilerplate code.

Key Components:
  - `Plotter`: The main class for creating plots. It offers methods like
    `plot_line`, `plot_scatter`, etc. (see `optiland.plotting.core.Plotter`).
  - `config`: Module for managing global plot settings such as figure size,
    font sizes, and default behaviors (see `optiland.plotting.config`).
  - `themes`: Module for defining and managing plot themes. Themes control
    colors, styles, and other visual aspects of plots (see
    `optiland.plotting.themes`).
  - `exceptions`: Defines custom exceptions for the plotting module, allowing
    for more specific error handling (see `optiland.plotting.exceptions`).

Basic Usage Example:
  Instantiate the `Plotter`, prepare some data, and call a plotting method:

  >>> from optiland.plotting import Plotter
  >>> plotter = Plotter()
  >>> x_data = [1, 2, 3, 4, 5]
  >>> y_data = [2, 4, 1, 3, 5]
  >>> plotter.plot_line(x_data, y_data, title="Simple Line Plot",
  ...                   xlabel="X-Axis", ylabel="Y-Axis")
  # The plot will be shown automatically if 'plot.show_on_draw' is True (default).

  To change theme:
  >>> plotter.set_theme('dark')
  >>> plotter.plot_scatter(x_data, y_data, title="Dark Theme Scatter Plot")

  To customize global settings:
  >>> from optiland.plotting import config
  >>> config.set_config('font.size_title', 20)
  >>> plotter.plot_line(x_data, y_data, title="Larger Title Plot")

  To get figure and axes objects for further customization:
  >>> fig, ax = plotter.plot_line(x_data, y_data, return_fig_ax=True)
  >>> if fig and ax:
  ...     ax.axvline(x=3, color='red', linestyle='--') # Add a vertical line
  ...     # import matplotlib.pyplot as plt # Required if plt.show() is needed
  ...     # Manually show if 'plot.show_on_draw' is False or if 'return_fig_ax' was True
  ...     # plt.show()

See the documentation for `Plotter`, `config`, and `themes` for more details.
"""

from . import config, exceptions, themes
from .core import Plotter
from .plot_configs import LegendConfig  # Added import

__all__ = [
    "Plotter",
    "LegendConfig",  # Added to __all__
    "config",
    "themes",
    "exceptions",
]

# Initialize default theme to ensure matplotlib settings are applied early
# This is a potential place to apply the default theme to matplotlib's rcParams
# if desired, e.g., by calling a function that updates plt.rcParams.
# For now, the theme application is done per plot.
# Example: themes.apply_theme_globally(themes.get_active_theme())
pass
