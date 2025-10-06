# tests/analysis/test_spot_vs_field.py
"""
Tests for the RmsSpotSizeVsField analysis tool.
"""
import matplotlib
import matplotlib.pyplot as plt
import pytest

import optiland.backend as be
from optiland import analysis
from optiland.samples.objectives import TripletTelescopeObjective

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def telescope_objective():
    """Provides a TripletTelescopeObjective instance for testing."""
    return TripletTelescopeObjective()


class TestSpotVsField:
    """
    Tests the RmsSpotSizeVsField analysis, which plots the RMS spot size as a
    function of the field height.
    """

    def test_rms_spot_size_vs_field_initialization(
        self, set_test_backend, telescope_objective
    ):
        """
        Tests the initialization of the analysis tool, verifying that the
        correct number of field points are generated.
        """
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        assert spot_vs_field.num_fields == 64
        assert be.array_equal(spot_vs_field._field[:, 1], be.linspace(0, 1, 64))

    def test_rms_spot_radius(self, set_test_backend, telescope_objective):
        """
        Tests that the spot size data is generated with the correct shape.
        """
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        spot_size = spot_vs_field._spot_size
        assert spot_size.shape == (
            64,
            len(telescope_objective.wavelengths.get_wavelengths()),
        )

    def test_view_spot_vs_field(self, set_test_backend, telescope_objective):
        """
        Tests the view method for generating a spot size vs. field plot.
        """
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        fig, ax = spot_vs_field.view()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_view_spot_vs_field_larger_fig(self, set_test_backend, telescope_objective):
        """
        Tests that the view method can accept a custom figure size.
        """
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        fig, ax = spot_vs_field.view(figsize=(12.4, 10))
        assert fig is not None
        assert ax is not None
        plt.close(fig)