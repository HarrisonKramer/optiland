# tests/analysis/test_yybar.py
"""
Tests for the YYbar analysis tool.
"""
import matplotlib
import matplotlib.pyplot as plt
import pytest

from optiland import analysis
from optiland.samples.objectives import TripletTelescopeObjective

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def telescope_objective():
    """Provides a TripletTelescopeObjective instance for testing."""
    return TripletTelescopeObjective()


class TestTelescopeTripletYYbar:
    """
    Tests the YYbar analysis, which plots the paraxial marginal and chief
    ray heights at each surface.
    """

    def test_view_yybar(self, set_test_backend, telescope_objective):
        """
        Tests the view method for generating a YY-bar plot.
        """
        yybar = analysis.YYbar(telescope_objective)
        fig, ax = yybar.view()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_view_yybar_larger_fig(self, set_test_backend, telescope_objective):
        """
        Tests that the view method can accept a custom figure size.
        """
        yybar = analysis.YYbar(telescope_objective)
        fig, ax = yybar.view(figsize=(12.4, 10))
        assert fig is not None
        assert ax is not None
        plt.close(fig)