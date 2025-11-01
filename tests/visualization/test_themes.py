"""Tests for the visualization themes module."""

import pytest
from matplotlib.colors import to_rgb

import matplotlib.pyplot as plt

from optiland.samples.simple import AsphericSinglet
from optiland.visualization.themes import (
    Theme,
    get_active_theme,
    set_theme,
    theme_context,
)
from optiland.visualization.palettes import light_palette


def test_set_theme():
    """Test setting a theme by name."""
    set_theme("dark")
    theme = get_active_theme()
    assert theme.name == "dark"
    set_theme("light")
    theme = get_active_theme()
    assert theme.name == "light"


def test_set_theme_object():
    """Test setting a theme with a Theme object."""
    custom_theme = Theme("custom", "A custom theme", light_palette)
    set_theme(custom_theme)
    theme = get_active_theme()
    assert theme.name == "custom"


def test_set_theme_not_found():
    """Test that setting a non-existent theme raises a ValueError."""
    with pytest.raises(ValueError):
        set_theme("non_existent_theme")


def test_theme_context():
    """Test the theme_context context manager."""
    set_theme("light")
    with theme_context("dark"):
        assert get_active_theme().name == "dark"
    assert get_active_theme().name == "light"


def test_optic_viewer_applies_theme(set_test_backend):
    """Test that the OpticViewer correctly applies the active theme."""
    optic = AsphericSinglet()

    set_theme("dark")
    fig, ax = optic.draw()

    dark_theme = get_active_theme()
    assert fig.get_facecolor()[:3] == to_rgb(dark_theme.parameters["figure.facecolor"])
    assert ax.get_facecolor()[:3] == to_rgb(dark_theme.parameters["axes.facecolor"])

    plt.close(fig)
