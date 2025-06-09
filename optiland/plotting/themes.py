"""Manages plotting themes for the Optiland library.

This module defines various visual themes that can be applied to plots,
allowing for consistent styling across different visualizations. It provides
functionality to set, get, and manage these themes.
"""

import cycler
import matplotlib.cm as cm  # Changed import for colormaps

from . import exceptions

THEMES = {
    "light": {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#f8f9fa",
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
        "axes.prop_cycle": cycler.cycler(color=cm.get_cmap("tab10").colors),
    },
    "dark": {
        "figure.facecolor": "#12151C",  # Subtle dark navy tint
        "axes.facecolor": "#1A1E26",  # Slight blue tint for contrast
        "axes.edgecolor": "#CCCCCC",
        "axes.labelcolor": "#E0E0E0",
        "axes.titlecolor": "#FFFFFF",
        "text.color": "#E0E0E0",
        "xtick.color": "#BBBBBB",
        "ytick.color": "#BBBBBB",
        "grid.color": "#444C56",  # Cooler grid lines
        "grid.alpha": 0.6,
        "lines.color": "#00BFFF",  # Deep Sky Blue (ignored if using prop_cycle)
        "lines.linewidth": 1.5,
        "patch.edgecolor": "#CCCCCC",
        "legend.facecolor": "#1A1E26",
        "legend.edgecolor": "#444C56",
        "legend.labelcolor": "#E0E0E0",
        # 3D specific
        "axes3d.facecolor": "#1A1E26",
        "axes3d.pane_color": "#20242C",
        "axes3d.grid_color": "#3A3F4B",
        "axes.prop_cycle": cycler.cycler(
            color=[
                "#FF6E54",  # bright orange-red
                "#6BFFB5",  # mint green
                "#FFD700",  # vivid gold
                "#56B4E9",  # high-contrast sky blue
                "#C77CFF",  # light violet
                "#00FA9A",  # medium spring green
                "#FF69B4",  # hot pink
                "#8BE9FD",  # soft neon cyan
                "#F08080",  # light coral
                "#A6CEE3",  # pastel blue (fallback)
            ]
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
