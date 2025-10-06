# tests/analysis/test_encircled_energy.py
"""
Tests for the EncircledEnergy analysis tool.
"""
import matplotlib
import matplotlib.pyplot as plt
import pytest

from optiland import analysis
from optiland.samples.objectives import CookeTriplet
from ..utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def cooke_triplet():
    """Provides a CookeTriplet instance for testing."""
    return CookeTriplet()


class TestCookeTripletEncircledEnergy:
    """
    Tests the EncircledEnergy analysis for the Cooke Triplet lens system.
    """

    def test_encircled_energy_centroid(self, set_test_backend, cooke_triplet):
        """
        Tests the calculation of the spot centroid.

        Note: Encircled energy calculations involve random sampling, so a
        tolerance is used for the assertion.
        """
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        centroid = encircled_energy.centroid()

        # Field 1
        assert_allclose(centroid[0][0], -8.207e-06, atol=1e-3, rtol=1e-3)
        assert_allclose(centroid[0][1], 1.989e-06, atol=1e-3, rtol=1e-3)

        # Field 2
        assert_allclose(centroid[1][0], 3.069e-05, atol=1e-3, rtol=1e-3)
        assert_allclose(centroid[1][1], 12.4213, atol=1e-3, rtol=1e-3)

        # Field 3
        assert_allclose(centroid[2][0], 3.163e-07, atol=1e-3, rtol=1e-3)
        assert_allclose(centroid[2][1], 18.1350, atol=1e-3, rtol=1e-3)

    def test_view_encircled_energy(self, set_test_backend, cooke_triplet):
        """
        Tests the view method for generating an encircled energy plot.
        """
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        fig, ax = encircled_energy.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_view_encircled_energy_larger_fig(self, set_test_backend, cooke_triplet):
        """
        Tests that the view method can accept a custom figure size.
        """
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        fig, ax = encircled_energy.view(figsize=(20, 10))
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)