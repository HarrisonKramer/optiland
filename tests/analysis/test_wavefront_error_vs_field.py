# tests/analysis/test_wavefront_error_vs_field.py
"""
Tests for the RmsWavefrontErrorVsField analysis tool.
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


class TestWavefrontErrorVsField:
    """
    Tests the RmsWavefrontErrorVsField analysis, which plots the RMS
    wavefront error as a function of the field height.
    """

    def test_rms_wave_init(self, set_test_backend, telescope_objective):
        """
        Tests the initialization of the analysis tool, verifying that the
        correct number of field points are generated.
        """
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        assert wavefront_error_vs_field.num_fields == 32
        assert be.array_equal(
            wavefront_error_vs_field._field[:, 1],
            be.linspace(0, 1, 32),
        )

    def test_rms_wave(self, set_test_backend, telescope_objective):
        """
        Tests that the wavefront error data is generated with the correct shape.
        """
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        wavefront_error = wavefront_error_vs_field._wavefront_error
        assert wavefront_error.shape == (
            32,
            len(telescope_objective.wavelengths.get_wavelengths()),
        )

    def test_view_wave(self, set_test_backend, telescope_objective):
        """
        Tests the view method for generating a wavefront error vs. field plot.
        """
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        fig, ax = wavefront_error_vs_field.view()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_view_wave_larger_fig(self, set_test_backend, telescope_objective):
        """
        Tests that the view method can accept a custom figure size.
        """
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        fig, ax = wavefront_error_vs_field.view(figsize=(12.4, 10))
        assert fig is not None
        assert ax is not None
        plt.close(fig)