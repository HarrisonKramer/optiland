# tests/psf/test_mmdft_psf.py
"""
Tests for the MMDFTPSF analysis tool in optiland.psf.
"""
from unittest.mock import patch
import matplotlib
import matplotlib.pyplot as plt
import optiland.backend as be
import pytest
from contextlib import nullcontext as does_not_raise

from optiland.psf import MMDFTPSF, FFTPSF
from optiland.samples.objectives import CookeTriplet
from ..utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing

@pytest.fixture
def make_mmdftpsf(set_test_backend):
    """
    A factory fixture to create an MMDFTPSF instance for a CookeTriplet with
    configurable parameters.
    """
    def _factory(
        field=(0, 0),
        wavelength=0.55,
        num_rays=128,
        image_size=128,
        pixel_pitch=None,
        tweak_optic=None,
    ):
        optic = CookeTriplet()
        if tweak_optic:
            tweak_optic(optic)
        return MMDFTPSF(
            optic, field, wavelength,
            num_rays=num_rays, image_size=image_size, pixel_pitch=pixel_pitch
        )
    return _factory

@pytest.fixture
def make_mmdftpsf_and_fftpsf(set_test_backend):
    """
    A factory fixture that creates both an FFTPSF and an MMDFTPSF instance
    with compatible parameters to test for agreement between the two methods.
    """
    def _factory(
        field=(0, 0),
        wavelength=0.55,
        num_rays=128,
        image_size=128,
        pixel_pitch=None,
        tweak_optic=None,
    ):
        optic = CookeTriplet()
        if tweak_optic:
            tweak_optic(optic)
        # Create FFTPSF first to determine the pixel pitch for MMDFTPSF
        fftpsf = FFTPSF(optic,
                        field,
                        wavelength,
                        num_rays=num_rays,
                        grid_size=image_size)
        dx = (
                fftpsf.wavelengths[0] *
                fftpsf._get_working_FNO() *
                (fftpsf.num_rays - 1) / fftpsf.grid_size
        )
        mmdftpsf = MMDFTPSF(optic,
                            field,
                            wavelength,
                            num_rays=fftpsf.num_rays,
                            image_size=fftpsf.grid_size,
                            pixel_pitch=dx)
        return fftpsf, mmdftpsf
    return _factory


class TestMMDFTPSF:
    """
    Tests the MMDFTPSF class, which calculates the Point Spread Function (PSF)
    using a Matrix M-DFT (Matrix Discrete Fourier Transform) method.
    """

    def test_initialization(self, make_mmdftpsf):
        """
        Tests that the MMDFTPSF class initializes correctly and that the
        resulting PSF has the expected shape.
        """
        mmdftpsf = make_mmdftpsf(image_size=1024)
        assert mmdftpsf.image_size == 1024
        assert mmdftpsf.psf.shape == (1024, 1024)

    @pytest.mark.parametrize(
        "num_rays,expected_pupil_sampling, expected_pixel_pitch",
        [
            (32, 32, 1.32622273171),
            (64, 45, 0.94119032573),
            (128, 64, 0.67380671047),
            (256, 90, 0.47594283517),
            (1024, 181, 0.24064525374),
        ],
    )
    def test_calcs_from_num_rays(self, make_mmdftpsf, num_rays, expected_pupil_sampling, expected_pixel_pitch):
        """
        Tests that when only `num_rays` is specified, the pupil sampling rate,
        image size, and pixel pitch are calculated correctly.
        """
        mmdftpsf = make_mmdftpsf(num_rays=num_rays, image_size=None)
        assert mmdftpsf.num_rays == expected_pupil_sampling
        assert mmdftpsf.image_size == 2 * num_rays
        assert_allclose(mmdftpsf.pixel_pitch, expected_pixel_pitch)

    @pytest.mark.parametrize(
        "pixel_pitch, expected_image_size",
        [(0.25, 1390), (0.50, 695), (1.00, 347), (2.00, 173)]
    )
    def test_calcs_from_pixel_pitch(self, make_mmdftpsf, pixel_pitch, expected_image_size):
        """
        Tests that when only `pixel_pitch` is specified, the image size is
        calculated correctly to satisfy the sampling theorem.
        """
        mmdftpsf = make_mmdftpsf(pixel_pitch=pixel_pitch, image_size=None)
        assert mmdftpsf.image_size == expected_image_size

    @pytest.mark.parametrize(
        "image_size, expected_pixel_pitch",
        [(128, 2.7166), (256, 1.3583), (512, 0.6791), (1024, 0.3395)]
    )
    def test_calcs_from_image_size(self, make_mmdftpsf, image_size, expected_pixel_pitch):
        """
        Tests that when only `image_size` is specified, the pixel pitch is
        calculated correctly.
        """
        mmdftpsf = make_mmdftpsf(image_size=image_size, pixel_pitch=None)
        assert_allclose(expected_pixel_pitch, mmdftpsf.pixel_pitch, atol=1e-4)

    def test_invalid_image_size(self, make_mmdftpsf):
        """
        Tests that a ValueError is raised if the specified image size is too
        small for the calculated required padding.
        """
        with pytest.raises(ValueError, match="Supplied image_size"):
            make_mmdftpsf(image_size=400, pixel_pitch=1)

    def test_strehl_ratio(self, make_mmdftpsf):
        """
        Tests the Strehl ratio calculation, which should always be between 0 and 1.
        """
        mmdftpsf = make_mmdftpsf(image_size=256)
        strehl_ratio = mmdftpsf.strehl_ratio()
        assert 0 <= strehl_ratio <= 1

    @pytest.mark.parametrize("projection, log", [("2d", False), ("3d", True)])
    def test_view(self, projection, log, make_mmdftpsf):
        """
        Tests the view method with different projections and scaling to ensure
        it runs without error and returns a valid plot.
        """
        mmdftpsf = make_mmdftpsf(field=(0, 1))
        fig, ax = mmdftpsf.view(projection=projection, log=log)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    @pytest.mark.parametrize(
        "num_rays, image_size",
        [(32, 64), (64, 128), (128, 256), (256, 512), (128, 128), (256, 256)]
    )
    def test_fft_agreement(self, make_mmdftpsf_and_fftpsf, num_rays, image_size):
        """
        Tests that the MMDFTPSF results agree with the FFTPSF results when
        the sampling conditions are made equivalent. This is a critical
        cross-validation test.
        """
        fftpsf, mmdftpsf = make_mmdftpsf_and_fftpsf(num_rays=num_rays, image_size=image_size)
        assert_allclose(fftpsf.psf, mmdftpsf.psf)
        assert_allclose(fftpsf.strehl_ratio(), mmdftpsf.strehl_ratio())