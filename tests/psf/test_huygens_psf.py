# tests/psf/test_huygens_psf.py
"""
Tests for the HuygensPSF analysis tool in optiland.psf.
"""
import matplotlib
import matplotlib.pyplot as plt
import pytest

from optiland import psf
from optiland.samples.objectives import CookeTriplet
from ..utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def cooke_triplet():
    """Provides a CookeTriplet instance for testing."""
    return CookeTriplet()


class TestHuygensPSF:
    """
    Tests the HuygensPSF class, which calculates the Point Spread Function (PSF)
    by propagating wavelets from the exit pupil to the image plane.
    """

    def test_init(self, set_test_backend, cooke_triplet):
        """
        Tests the initialization of the HuygensPSF analysis tool, verifying
        that the optic and other parameters are set correctly.
        """
        psf_calculator = psf.HuygensPSF(cooke_triplet, num_rays=64)
        assert psf_calculator.optic == cooke_triplet
        assert psf_calculator.num_rays == 64
        assert psf_calculator.image_size == 256

    def test_generate_data(self, set_test_backend, cooke_triplet):
        """
        Tests the data generation process by comparing calculated PSF values
        against known reference values for a specific field and wavelength.
        """
        psf_calculator = psf.HuygensPSF(cooke_triplet, num_rays=32)
        data = psf_calculator.data

        # Check data for the first field and first wavelength
        field_data = data[0]
        wavelength_data = field_data[0]
        # Check some specific PSF values against known results
        assert_allclose(wavelength_data["psf"][128, 128], 1.0)  # Peak should be 1.0
        assert_allclose(wavelength_data["psf"][128, 135], 0.00488, atol=1e-5)
        assert_allclose(wavelength_data["psf"][120, 124], 0.00015, atol=1e-5)
        # Check that the coordinate arrays are correct
        assert_allclose(wavelength_data["x"][0], -0.015, atol=1e-3)
        assert_allclose(wavelength_data["y"][-1], 0.015, atol=1e-3)

    def test_view(self, set_test_backend, cooke_triplet):
        """
        Tests the view method for generating a PSF plot, ensuring it returns
        a valid matplotlib Figure and Axes.
        """
        psf_calculator = psf.HuygensPSF(cooke_triplet)
        fig, ax = psf_calculator.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_view_with_customizations(self, set_test_backend, cooke_triplet):
        """
        Tests that the view method can accept custom plotting parameters,
        such as figure size and color map.
        """
        psf_calculator = psf.HuygensPSF(cooke_triplet)
        fig, ax = psf_calculator.view(
            figsize=(10, 8),
            cmap="viridis",
            scale="log",
            num_levels=20,
            add_colorbar=True,
        )
        assert fig is not None
        assert ax is not None
        assert fig.get_size_inches()[0] == 10
        plt.close(fig)