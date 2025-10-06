# tests/analysis/test_field_curvature.py
"""
Tests for the FieldCurvature analysis tool.
"""
import matplotlib
import matplotlib.pyplot as plt
import pytest

from optiland import analysis
from optiland.samples.objectives import TripletTelescopeObjective
from ..utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def telescope_objective():
    """Provides a TripletTelescopeObjective instance for testing."""
    return TripletTelescopeObjective()


class TestTelescopeTripletFieldCurvature:
    """
    Tests the FieldCurvature analysis for the Triplet Telescope Objective.
    """

    def test_field_curvature_init(self, set_test_backend, telescope_objective):
        """
        Tests the initialization of the FieldCurvature analysis tool,
        including default and custom parameters.
        """
        # Test with default parameters
        fc = analysis.FieldCurvature(telescope_objective)
        assert fc.optic == telescope_objective
        assert fc.wavelengths == telescope_objective.wavelengths.get_wavelengths()
        assert fc.num_points == 128

        # Test with custom parameters
        fc_custom = analysis.FieldCurvature(
            telescope_objective, wavelengths=[0.55], num_points=256
        )
        assert fc_custom.wavelengths == [0.55]
        assert fc_custom.num_points == 256

    def test_field_curvature_view(self, set_test_backend, telescope_objective):
        """
        Tests the view method for generating a field curvature plot.
        """
        field_curvature = analysis.FieldCurvature(telescope_objective)
        fig, ax = field_curvature.view()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_field_curvature_generate_data(self, set_test_backend, telescope_objective):
        """
        Tests the data generation by comparing calculated tangential and
        sagittal focus positions against known values.
        """
        fc = analysis.FieldCurvature(telescope_objective)
        # Check tangential focus values
        assert_allclose(fc.data[0][0][89], -0.001306, atol=1e-5)
        assert_allclose(fc.data[0][0][40], 0.029698, atol=1e-5)
        # Check sagittal focus values
        assert_allclose(fc.data[1][1][55], -0.004469, atol=1e-5)
        assert_allclose(fc.data[1][1][19], 0.000800, atol=1e-5)
        # Check Petzval surface values
        assert_allclose(fc.data[2][1][62], 0.059485, atol=1e-5)
        assert_allclose(fc.data[2][0][0], 0.067070, atol=1e-5)