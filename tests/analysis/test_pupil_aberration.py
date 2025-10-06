# tests/analysis/test_pupil_aberration.py
"""
Tests for the PupilAberration analysis tool.
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


class TestPupilAberration:
    """
    Tests the PupilAberration analysis, which plots the distortion of the
    pupil.
    """

    def test_initialization(self, set_test_backend, telescope_objective):
        """
        Tests the initialization of the analysis tool, including default
        and custom parameters.
        """
        pupil_ab = analysis.PupilAberration(telescope_objective)
        assert pupil_ab.optic == telescope_objective
        assert pupil_ab.fields == [(0.0, 0.0), (0.0, 0.7), (0.0, 1.0)]
        assert pupil_ab.wavelengths == [0.4861, 0.5876, 0.6563]
        assert pupil_ab.num_points == 257  # Default 256 becomes odd

    def test_generate_data(self, set_test_backend, telescope_objective):
        """
        Tests that the data generation produces a dictionary with the
        expected structure and keys.
        """
        pupil_ab = analysis.PupilAberration(telescope_objective)
        data = pupil_ab._generate_data()
        assert "Px" in data
        assert "Py" in data
        assert "(0.0, 0.0)" in data
        assert "(0.0, 0.7)" in data
        assert "(0.0, 1.0)" in data
        assert "0.4861" in data["(0.0, 0.0)"]
        assert "x" in data["(0.0, 0.0)"]["0.4861"]
        assert "y" in data["(0.0, 0.0)"]["0.4861"]

    def test_view(self, set_test_backend, telescope_objective):
        """
        Tests the view method for generating a pupil aberration plot.
        """
        pupil_ab = analysis.PupilAberration(telescope_objective)
        fig, axes = pupil_ab.view()
        assert fig is not None
        assert axes is not None
        plt.close(fig)