from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import optiland.backend as be
import pytest

from optiland.psf import FFTPSF
from optiland.samples.objectives import CookeTriplet

matplotlib.use("Agg")  # use non-interactive backend for testing


def test_initialization():
    optic = CookeTriplet()
    field = (0, 0)
    wavelength = 0.57
    num_rays = 128
    grid_size = 1024

    fftpsf = FFTPSF(optic, field, wavelength, num_rays, grid_size)

    assert fftpsf.grid_size == grid_size
    assert len(fftpsf.pupils) == 1
    assert fftpsf.psf.shape == (grid_size, grid_size)


def test_strehl_ratio():
    optic = CookeTriplet()
    field = (0, 0)
    wavelength = 0.52
    num_rays = 128
    grid_size = 256

    fftpsf = FFTPSF(optic, field, wavelength, num_rays, grid_size)

    strehl_ratio = fftpsf.strehl_ratio()
    assert isinstance(strehl_ratio, float)
    assert 0 <= strehl_ratio <= 1


@patch("matplotlib.pyplot.show")
def test_view_2d(mock_show):
    optic = CookeTriplet()
    field = (0, 1)
    wavelength = 0.57
    num_rays = 128
    grid_size = 128

    fftpsf = FFTPSF(optic, field, wavelength, num_rays, grid_size)
    fftpsf.view(projection="2d")
    mock_show.assert_called_once()
    plt.close()


@patch("matplotlib.pyplot.show")
def test_view_3d(mock_show):
    optic = CookeTriplet()
    field = (0, 1)
    wavelength = 0.58
    num_rays = 128
    grid_size = 128

    fftpsf = FFTPSF(optic, field, wavelength, num_rays, grid_size)
    fftpsf.view(projection="3d")
    mock_show.assert_called_once()
    plt.close()


def test_find_bounds():
    optic = CookeTriplet()
    field = (0, 0)
    wavelength = 0.46
    num_rays = 128
    grid_size = 128

    fftpsf = FFTPSF(optic, field, wavelength, num_rays, grid_size)

    min_x, min_y, max_x, max_y = fftpsf._find_bounds(threshold=0.05)
    assert min_x >= 0
    assert min_y >= 0
    assert max_x <= grid_size
    assert max_y <= grid_size


@patch("matplotlib.pyplot.show")
def test_view_log_2d(mock_show):
    optic = CookeTriplet()
    field = (0, 1)
    wavelength = 0.66
    num_rays = 128
    grid_size = 128

    fftpsf = FFTPSF(optic, field, wavelength, num_rays, grid_size)
    fftpsf.view(projection="2d", log=True)
    mock_show.assert_called_once()
    plt.close()


@patch("matplotlib.pyplot.show")
def test_view_log_3d(mock_show):
    optic = CookeTriplet()
    field = (0, 1)
    wavelength = 0.65
    num_rays = 128
    grid_size = 128

    fftpsf = FFTPSF(optic, field, wavelength, num_rays, grid_size)
    fftpsf.view(projection="3d", log=True)
    mock_show.assert_called_once()
    plt.close()


def test_view_invalid_projection():
    optic = CookeTriplet()
    field = (0, 1)
    wavelength = 0.55
    num_rays = 128
    grid_size = 128

    fftpsf = FFTPSF(optic, field, wavelength, num_rays, grid_size)
    with pytest.raises(ValueError):
        fftpsf.view(projection="invalid", log=True)


def test_get_units_finite_obj():
    optic = CookeTriplet()
    # make object distance large, but not infinite
    optic.surface_group.surfaces[0].geometry.cs.z = 1e6
    field = (0, 1)
    wavelength = 0.55
    num_rays = 128
    grid_size = 128

    fftpsf = FFTPSF(optic, field, wavelength, num_rays, grid_size)
    image = be.zeros((128, 128))
    x, y = fftpsf._get_psf_units(image)
    assert be.isclose(x, 352.01567006276366)
    assert be.isclose(y, 352.01567006276366)


def test_psf_log_tick_formatter():
    optic = CookeTriplet()
    field = (0, 1)
    wavelength = 0.55
    num_rays = 128
    grid_size = 128

    fftpsf = FFTPSF(optic, field, wavelength, num_rays, grid_size)
    assert fftpsf._log_tick_formatter(10) == "$10^{10}$"
    assert fftpsf._log_tick_formatter(1) == "$10^{1}$"
    assert fftpsf._log_tick_formatter(0) == "$10^{0}$"
    assert fftpsf._log_tick_formatter(-1) == "$10^{-1}$"
    assert fftpsf._log_tick_formatter(-10) == "$10^{-10}$"


def test_interpolate_zoom_factor_one():
    optic = CookeTriplet()
    field = (0, 1)
    wavelength = 0.55
    num_rays = 128
    grid_size = 128

    fftpsf = FFTPSF(optic, field, wavelength, num_rays, grid_size)
    assert fftpsf._interpolate_psf(fftpsf.psf) is fftpsf.psf


def test_large_threshold():
    optic = CookeTriplet()
    field = (0, 1)
    wavelength = 0.55
    num_rays = 128
    grid_size = 128

    fftpsf = FFTPSF(optic, field, wavelength, num_rays, grid_size)
    min_x, min_y, max_x, max_y = fftpsf._find_bounds(threshold=100)
    assert min_x == 0
    assert min_y == 0
    assert max_x == grid_size
    assert max_y == grid_size
