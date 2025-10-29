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
    "ray_cycle": [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
}

# Base palette for the 'dark' theme
dark_palette = {
    "background": "#121212",
    "axis": "#E0E0E0",
    "text": "#E0E0E0",
    "grid": "#444444",
    "lens": "#555555",
    "ray_cycle": [
        "#A9C2DB",
        "#8AB9FF",
        "#FF8A8A",
        "#FFD18A",
        "#8AFFB5",
        "#B58AFF",
    ],
}

# Solarized light palette
solarized_light_palette = {
    "background": "#fdf6e3",
    "axis": "#657b83",
    "text": "#586e75",
    "grid": "#eee8d5",
    "lens": "#93a1a1",
    "ray_cycle": [
        "#268bd2",
        "#2aa198",
        "#859900",
        "#d33682",
        "#cb4b16",
        "#6c71c4",
    ],
}

# Solarized dark palette
solarized_dark_palette = {
    "background": "#002b36",
    "axis": "#839496",
    "text": "#93a1a1",
    "grid": "#073642",
    "lens": "#586e75",
    "ray_cycle": [
        "#2aa198",
        "#268bd2",
        "#859900",
        "#d33682",
        "#cb4b16",
        "#6c71c4",
    ],
}
