# tests/analysis/test_distortion.py
"""
Tests for the Distortion and GridDistortion analysis tools.
"""
import matplotlib
import matplotlib.pyplot as plt
import pytest

import optiland.backend as be
from optiland import analysis
from optiland.samples.objectives import TripletTelescopeObjective
from ..utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def telescope_objective():
    """Provides a TripletTelescopeObjective instance for testing."""
    return TripletTelescopeObjective()


class TestTelescopeTripletDistortion:
    """
    Tests the Distortion analysis for the Triplet Telescope Objective.
    """

    def test_distortion_values(self, set_test_backend, telescope_objective):
        """
        Tests the standard distortion calculation against known values.
        """
        dist = analysis.Distortion(telescope_objective)
        assert_allclose(dist.data[0][0], 0.0, atol=1e-9)
        assert_allclose(dist.data[0][-1], 0.0059505, atol=1e-5)
        assert_allclose(dist.data[1][-1], 0.0057863, atol=1e-5)
        assert_allclose(dist.data[2][-1], 0.0057203, atol=1e-5)

    def test_f_theta_distortion(self, set_test_backend, telescope_objective):
        """
        Tests the f-theta distortion calculation against known values.
        """
        dist = analysis.Distortion(telescope_objective, distortion_type="f-theta")
        assert_allclose(dist.data[0][-1], 0.016106, atol=1e-5)
        assert_allclose(dist.data[1][-1], 0.015942, atol=1e-5)
        assert_allclose(dist.data[2][-1], 0.015876, atol=1e-5)

    def test_invalid_distortion_type(self, set_test_backend, telescope_objective):
        """
        Tests that initializing with an invalid distortion type raises a
        ValueError.
        """
        with pytest.raises(ValueError):
            analysis.Distortion(telescope_objective, distortion_type="invalid")

    def test_view_distortion(self, set_test_backend, telescope_objective):
        """
        Tests the view method for generating a distortion plot.
        """
        dist = analysis.Distortion(telescope_objective)
        fig, ax = dist.view(figsize=(12.4, 10))
        assert fig is not None
        assert ax is not None
        plt.close(fig)


class TestTelescopeTripletGridDistortion:
    """
    Tests the GridDistortion analysis for the Triplet Telescope Objective.
    """

    def test_grid_distortion_values(self, set_test_backend, telescope_objective):
        """
        Tests the standard grid distortion calculation against known values.
        """
        dist = analysis.GridDistortion(telescope_objective)
        assert_allclose(dist.data["max_distortion"], 0.0057857, atol=1e-5)
        assert dist.data["xr"].shape == (10, 10)
        assert_allclose(dist.data["xr"][0, 0], -1.23426, atol=1e-5)
        assert_allclose(dist.data["yp"][1, 5], -0.95990, atol=1e-5)

    def test_f_theta_distortion(self, set_test_backend, telescope_objective):
        """
        Tests the f-theta grid distortion calculation against known values.
        """
        dist = analysis.GridDistortion(telescope_objective, distortion_type="f-theta")
        assert_allclose(dist.data["max_distortion"], 0.010863, atol=1e-5)
        assert_allclose(dist.data["xp"][0, 2], -0.68562, atol=1e-5)
        assert_allclose(dist.data["yp"][-1, 0], 1.23412, atol=1e-5)

    def test_invalid_distortion_type(self, set_test_backend, telescope_objective):
        """
        Tests that initializing with an invalid distortion type raises a
        ValueError.
        """
        with pytest.raises(ValueError):
            analysis.GridDistortion(telescope_objective, distortion_type="invalid")

    def test_view_grid_distortion(self, set_test_backend, telescope_objective):
        """
        Tests the view method for generating a grid distortion plot.
        """
        dist = analysis.GridDistortion(telescope_objective)
        fig, ax = dist.view(figsize=(12.4, 10))
        assert fig is not None
        assert ax is not None
        plt.close(fig)