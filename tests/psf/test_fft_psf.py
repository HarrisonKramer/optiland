# tests/psf/test_fft_psf.py
"""
Tests for the FFTPSF analysis tool in optiland.psf.
"""
from unittest.mock import patch
import matplotlib
import matplotlib.pyplot as plt
import optiland.backend as be
import pytest
from contextlib import nullcontext as does_not_raise

from optiland.psf import FFTPSF
from optiland.psf.fft import calculate_grid_size
from optiland.samples.objectives import CookeTriplet
from ..utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def make_fftpsf(set_test_backend):
    """
    A factory fixture to create an FFTPSF instance for a CookeTriplet with
    configurable parameters.
    """
    def _factory(field=(0, 0), wavelength=0.55, num_rays=128, grid_size=128, tweak_optic=None):
        optic = CookeTriplet()
        if tweak_optic:
            tweak_optic(optic)
        return FFTPSF(optic, field, wavelength, num_rays, grid_size)
    return _factory


class TestFFTPSF:
    """
    Tests the FFTPSF class, which calculates the Point Spread Function (PSF)
    by taking the Fast Fourier Transform of the pupil function.
    """

    def test_initialization(self, make_fftpsf):
        """
        Tests that the FFTPSF class initializes correctly and that the
        resulting PSF has the expected shape.
        """
        fftpsf = make_fftpsf(grid_size=1024)
        assert fftpsf.grid_size == 1024
        assert len(fftpsf.pupils) == 1
        assert fftpsf.psf.shape == (1024, 1024)

    @pytest.mark.parametrize(
        "num_rays,expected_pupil_sampling",
        [(32, 32), (64, 45), (128, 64), (256, 90), (1024, 181)]
    )
    def test_calculate_grid_size(self, num_rays, expected_pupil_sampling):
        """
        Tests the helper function that determines the pupil sampling rate
        based on the desired number of rays in the FFT grid.
        """
        sampling, _ = calculate_grid_size(num_rays)
        assert sampling == expected_pupil_sampling

    def test_invalid_grid_size(self, make_fftpsf):
        """
        Tests that a ValueError is raised if the specified grid size is smaller
        than the number of rays.
        """
        with pytest.raises(ValueError, match="must be greater than or equal to"):
            make_fftpsf(grid_size=63, num_rays=64)

    def test_strehl_ratio(self, make_fftpsf):
        """
        Tests the Strehl ratio calculation, which should always be between 0 and 1.
        """
        fftpsf = make_fftpsf(grid_size=256)
        strehl_ratio = fftpsf.strehl_ratio()
        assert 0 <= strehl_ratio <= 1

    @pytest.mark.parametrize("projection, log", [("2d", False), ("3d", False), ("2d", True), ("3d", True)])
    def test_view(self, projection, log, make_fftpsf):
        """
        Tests the view method with different projections and scaling to ensure
        it runs without error and returns a valid plot.
        """
        fftpsf = make_fftpsf(field=(0, 1))
        fig, ax = fftpsf.view(projection=projection, log=log)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_view_invalid_projection(self, make_fftpsf):
        """
        Tests that an invalid projection type raises a ValueError.
        """
        fftpsf = make_fftpsf(field=(0, 1))
        with pytest.raises(ValueError):
            fftpsf.view(projection="invalid")

    def test_get_units_finite_obj(self, make_fftpsf):
        """
        Tests the calculation of PSF units for a finite object distance.
        """
        def tweak(optic):
            optic.surface_group.surfaces[0].geometry.cs.z = -be.array(1e6)

        fftpsf = make_fftpsf(field=(0, 1), tweak_optic=tweak)
        image = be.zeros((128, 128))
        x, y = fftpsf._get_psf_units(image)
        assert_allclose(x, 382.8276, atol=1e-4)
        assert_allclose(y, 382.8276, atol=1e-4)

    def test_invalid_working_FNO(self, make_fftpsf):
        """
        Tests that a ValueError is raised if the working F-number is invalid
        (e.g., due to extreme object distance).
        """
        def tweak(optic):
            optic.surface_group.surfaces[0].geometry.cs.z = -be.array(1e100)

        fftpsf = make_fftpsf(field=(0, 1), tweak_optic=tweak)
        with pytest.raises(ValueError):
            fftpsf.view()

    def test_large_threshold(self, make_fftpsf):
        """
        Tests the `_find_bounds` method with a threshold so high that no pixels
        are included, ensuring it returns the full image dimensions.
        """
        fftpsf = make_fftpsf(field=(0, 1))
        psf = be.to_numpy(fftpsf.psf)
        min_x, min_y, max_x, max_y = fftpsf._find_bounds(psf, threshold=100)
        assert min_x == 0
        assert min_y == 0
        assert max_x == 128
        assert max_y == 128