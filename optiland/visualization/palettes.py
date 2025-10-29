"""Palettes Module

This module defines color palettes for Optiland's visualization themes.
It provides a structured way to manage colors, ensuring consistency
and ease of customization. The palettes are designed to be aesthetically
pleasing and adaptable to different visualization contexts, such as 'light'
and 'dark' modes.

Each palette is a dictionary containing specific color definitions for various
plot elements, including background, axes, text, grid, lenses, and rays.
This modular approach allows for easy extension with new palettes and ensures
that themes can be built on a consistent color foundation.

Manuel Fragata Mendes, June 2025
"""

# Base palette for the 'light' theme
from __future__ import annotations

light_palette = {
    "background": "#FFFFFF",
    "axis": "#333333",
    "text": "#333333",
    "grid": "#CCCCCC",
    "lens": "#E0E0E0",
    "ray": "#1f77b4",
}

# Base palette for the 'dark' theme
dark_palette = {
    "background": "#121212",
    "axis": "#E0E0E0",
    "text": "#E0E0E0",
    "grid": "#444444",
    "lens": "#555555",
    "ray": "#A9C2DB",
}

# Solarized light palette
solarized_light_palette = {
    "background": "#fdf6e3",
    "axis": "#657b83",
    "text": "#586e75",
    "grid": "#eee8d5",
    "lens": "#93a1a1",
    "ray": "#268bd2",
}

# Solarized dark palette
solarized_dark_palette = {
    "background": "#002b36",
    "axis": "#839496",
    "text": "#93a1a1",
    "grid": "#073642",
    "lens": "#586e75",
    "ray": "#2aa198",
}
