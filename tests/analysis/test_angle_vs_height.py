# tests/analysis/test_angle_vs_height.py
"""
Tests for the AngleVsHeight analysis tool.
"""
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pytest

from optiland import analysis
from optiland.samples.objectives import CookeTriplet

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def cooke_triplet():
    """Provides a CookeTriplet instance for testing."""
    return CookeTriplet()


def test_angle_vs_height_initialization(set_test_backend, cooke_triplet):
    """
    Tests the initialization of the AngleVsHeight analysis tool, verifying
    that the optic and number of points are set correctly.
    """
    avh = analysis.AngleVsHeight(cooke_triplet, num_points=50)
    assert avh.optic == cooke_triplet
    assert avh.num_points == 50
    assert len(avh.data) > 0


def test_angle_vs_height_data_generation(set_test_backend, cooke_triplet):
    """
    Tests that the data generation produces a dictionary with the expected
    keys ('angle', 'height') and that the data arrays have the correct length.
    """
    avh = analysis.AngleVsHeight(cooke_triplet, num_points=10)
    assert "angle" in avh.data
    assert "height" in avh.data
    assert len(avh.data["angle"]) == 10
    assert len(avh.data["height"]) == 10


@patch("matplotlib.pyplot.show")
def test_angle_vs_height_view(mock_show, set_test_backend, cooke_triplet):
    """
    Tests the view method for generating an angle vs. height plot, ensuring
    it returns a valid matplotlib Figure and Axes.
    """
    avh = analysis.AngleVsHeight(cooke_triplet)
    fig, ax = avh.view()
    assert fig is not None
    assert ax is not None
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_angle_vs_height_view_with_figsize(set_test_backend, cooke_triplet):
    """
    Tests that the view method can accept a custom figure size.
    """
    avh = analysis.AngleVsHeight(cooke_triplet)
    fig, ax = avh.view(figsize=(10, 8))
    assert fig.get_size_inches()[0] == 10
    assert fig.get_size_inches()[1] == 8
    plt.close(fig)