from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import optiland.backend as be
import pytest

from optiland.psf import FFTPSF
from optiland.samples.objectives import CookeTriplet
from .utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def make_fftpsf(set_test_backend):
    def _factory(
        field=(0, 0),
        wavelength=0.55,
        num_rays=128,
        grid_size=128,
        tweak_optic=None,
    ):
        optic = CookeTriplet()
        if tweak_optic:
            tweak_optic(optic)
        return FFTPSF(optic, field, wavelength, num_rays, grid_size)
    return _factory


def test_initialization(make_fftpsf):
    fftpsf = make_fftpsf(grid_size=1024)
    assert fftpsf.grid_size == 1024
    assert len(fftpsf.pupils) == 1
    assert fftpsf.psf.shape == (1024, 1024)


def test_strehl_ratio(make_fftpsf):
    fftpsf = make_fftpsf(grid_size=256)
    strehl_ratio = fftpsf.strehl_ratio()
    assert 0 <= strehl_ratio <= 1


@patch("matplotlib.pyplot.show")
def test_view_2d(mock_show, make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))
    fftpsf.view(projection="2d")
    mock_show.assert_called_once()
    plt.close()


@patch("matplotlib.pyplot.show")
def test_view_3d(mock_show, make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))
    fftpsf.view(projection="3d")
    mock_show.assert_called_once()
    plt.close()


def test_find_bounds(make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))

    min_x, min_y, max_x, max_y = fftpsf._find_bounds(threshold=0.05)
    assert min_x >= 0
    assert min_y >= 0
    assert max_x <= 128
    assert max_y <= 128


@patch("matplotlib.pyplot.show")
def test_view_log_2d(mock_show, make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))
    fftpsf.view(projection="2d", log=True)
    mock_show.assert_called_once()
    plt.close()


@patch("matplotlib.pyplot.show")
def test_view_log_3d(mock_show, make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))
    fftpsf.view(projection="3d", log=True)
    mock_show.assert_called_once()
    plt.close()


def test_view_invalid_projection(make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))
    with pytest.raises(ValueError):
        fftpsf.view(projection="invalid", log=True)


def test_get_units_finite_obj(make_fftpsf):
    def tweak(optic):
        optic.surface_group.surfaces[0].geometry.cs.z = be.array(1e6)

    fftpsf = make_fftpsf(field=(0, 1), tweak_optic=tweak)
    image = be.zeros((128, 128))
    x, y = fftpsf._get_psf_units(image)
    assert_allclose(x, 352.01567006276366)
    assert_allclose(y, 352.01567006276366)


def test_psf_log_tick_formatter(make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))
    assert fftpsf._log_tick_formatter(10) == "$10^{10}$"
    assert fftpsf._log_tick_formatter(1) == "$10^{1}$"
    assert fftpsf._log_tick_formatter(0) == "$10^{0}$"
    assert fftpsf._log_tick_formatter(-1) == "$10^{-1}$"
    assert fftpsf._log_tick_formatter(-10) == "$10^{-10}$"


def test_interpolate_zoom_factor_one(make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))
    assert fftpsf._interpolate_psf(fftpsf.psf) is fftpsf.psf


def test_large_threshold(make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))
    psf = be.to_numpy(fftpsf.psf)
    min_x, min_y, max_x, max_y = fftpsf._find_bounds(psf, threshold=100)
    assert min_x == 0
    assert min_y == 0
    assert max_x == 128
    assert max_y == 128
