"""Manages plotting themes for the Optiland library.

This module defines various visual themes that can be applied to plots,
allowing for consistent styling across different visualizations. It provides
functionality to set, get, and manage these themes.
"""

import cycler
import matplotlib.colormaps

from . import exceptions

THEMES = {
    "light": {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#F0F0F0",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "axes.titlecolor": "#333333",
        "text.color": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.color": "#D0D0D0",
        "grid.alpha": 0.8,
        "lines.color": "#007ACC",  # Primary line color
        "lines.linewidth": 1.5,  # From config, but can be themed
        "patch.edgecolor": "#333333",
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#CCCCCC",
        # Matplotlib uses 'text.color' for legend text by default
        "legend.labelcolor": "#333333",
        # but specific control might be needed.
        # Using 'legend.labelcolor' as a custom key for now if needed.
        # For direct rcParams, many legend props are under `legend.*`
        # 3D specific
        "axes3d.facecolor": "#F0F0F0",  # Similar to 2D axes
        "axes3d.pane_color": "#EAEAF2",  # Slightly different for depth perception
        "axes3d.grid_color": "#C0C0C0",  # Slightly lighter than 2D grid
        "axes.prop_cycle": cycler.cycler(color=matplotlib.colormaps["tab10"].colors),
    },
    "dark": {
        "figure.facecolor": "#1E1E1E",
        "axes.facecolor": "#2D2D2D",
        "axes.edgecolor": "#CCCCCC",
        "axes.labelcolor": "#E0E0E0",  # Brighter than pure CCCCC
        "axes.titlecolor": "#FFFFFF",  # Pure white for titles
        "text.color": "#E0E0E0",
        "xtick.color": "#CCCCCC",
        "ytick.color": "#CCCCCC",
        "grid.color": "#555555",
        "grid.alpha": 0.7,
        "lines.color": "#00A0FF",  # Bright blue for dark theme
        "lines.linewidth": 1.5,
        "patch.edgecolor": "#CCCCCC",
        "legend.facecolor": "#2D2D2D",
        "legend.edgecolor": "#555555",
        "legend.labelcolor": "#E0E0E0",  # Matching general text color
        # 3D specific
        "axes3d.facecolor": "#2D2D2D",  # Similar to 2D axes
        "axes3d.pane_color": "#252525",  # Slightly different for depth perception
        "axes3d.grid_color": "#484848",  # Slightly lighter than 2D grid
        "axes.prop_cycle": cycler.cycler(
            color=[
                "#80B1D3",
                "#FFED6F",
                "#B3DE69",
                "#FCCDE5",
                "#BC80BD",
                "#CCEBC5",
                "#FFB347",
                "#FDB462",
                "#BEBADA",
                "#FB8072",
            ],
        ),
    },
}

_active_theme: str = "light"


def set_active_theme(theme_name: str):
    """Sets the active plotting theme.

    Args:
      theme_name: The name of the theme to activate (e.g., 'light', 'dark').

    Raises:
      exceptions.ThemeNotFoundError: If the `theme_name` is not a defined theme.

    """
    global _active_theme
    if theme_name not in THEMES:
        raise exceptions.ThemeNotFoundError(theme_name, list(THEMES.keys()))
    _active_theme = theme_name


def get_active_theme() -> str:
    """Returns the name of the currently active theme.

    Returns:
      The name of the active theme.

    """
    return _active_theme


def get_theme_value(key: str, theme_name: str = None):
    """Retrieves a specific value from a theme's configuration.

    If `theme_name` is provided, the value is retrieved from that theme.
    Otherwise, it's retrieved from the currently active theme.

    Args:
      key: The configuration key to retrieve (e.g., 'figure.facecolor').
      theme_name: Optional; the name of the theme to get the value from.
          If None, uses the active theme.

    Returns:
      The value associated with the key in the specified or active theme.

    Raises:
      exceptions.ThemeNotFoundError: If the specified `theme_name` is not valid,
                                     or if the `key` is not found in the theme.

    """
    current_theme_name = theme_name or _active_theme

    if current_theme_name not in THEMES:
        raise exceptions.ThemeNotFoundError(current_theme_name, list(THEMES.keys()))

    theme_dict = THEMES[current_theme_name]
    if key not in theme_dict:
        # Consider if a more general ConfigurationError is better if keys
        # might not be theme-specific.
        # For now, ThemeNotFoundError implies the key is missing from *this*
        # theme's definition.
        raise exceptions.ThemeNotFoundError(
            f"Key '{key}' not found in theme '{current_theme_name}'.",
            available_themes=list(THEMES.keys()),  # Pass available themes for context
        )
    return theme_dict[key]


def get_active_theme_dict() -> dict:
    """Returns the complete dictionary of the currently active theme.

    This provides all settings defined for the active theme.

    Returns:
      A dictionary containing all key-value pairs for the active theme.

    """
    return THEMES[_active_theme].copy()  # Return a copy to prevent modification


def list_themes() -> list[str]:
    """Returns a list of available theme names.

    Returns:
        A list of strings, where each string is a name of an available theme.

    """
    return list(THEMES.keys())
