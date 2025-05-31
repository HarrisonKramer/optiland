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


@pytest.mark.parametrize(
    "projection, log",
    [
        ("2d", False),
        ("3d", False),
        ("2d", True),
        ("3d", True),
    ],
)
@patch("matplotlib.pyplot.show")
def test_view(mock_show, projection, log, make_fftpsf, set_test_backend):
    # Skip for torch since view isn't implemented there
    fftpsf = make_fftpsf(field=(0, 1))
    fftpsf.view(projection=projection, log=log)
    mock_show.assert_called_once()
    plt.close("all")


def test_find_bounds(make_fftpsf):
    fftpsf = make_fftpsf(field=(0, 1))

    psf = be.to_numpy(fftpsf.psf)
    min_x, min_y, max_x, max_y = fftpsf._find_bounds(psf, threshold=0.05)
    assert min_x >= 0
    assert min_y >= 0
    assert max_x <= 128
    assert max_y <= 128


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


# --- Tests for _get_normalization logic ---

# Helper function to calculate expected normalization for a given binary pupil mask
def calculate_expected_norm_for_test(binary_pupil_mask, num_pupils_in_list=1):
    # Ensure pupil_mask is complex for fft2, matching _get_normalization's P_nom type before FFT
    complex_pupil_mask = be.astype(binary_pupil_mask, be.complex64)
    # The FFT is performed on the padded pupil in the actual code,
    # but _get_normalization uses the unpadded self.pupils[0] to create P_nom,
    # then pads P_nom for its own FFT.
    # Here, we assume binary_pupil_mask is already at the correct size (like P_nom before padding).
    # For simplicity in this helper, we'll assume it's not padded yet, matching how P_nom is handled
    # up to the point of its own FFT.
    # In _get_normalization, P_nom is derived from self.pupils[0] (size num_rays x num_rays)
    # then it's padded for FFT. This helper should conceptually do the same for an apples-to-apples comparison
    # with the *logic* of creating binary P_nom and its FFT peak.
    # However, fftpsf._get_normalization itself handles padding internally for its P_nom.
    # So, this helper should just calculate what be.max(be.real(fftshift(fft2(P_nom_binary)))) would be.

    amp_norm = be.fft.fftshift(be.fft.fft2(complex_pupil_mask))
    psf_norm = amp_norm * be.conj(amp_norm)
    return be.max(be.real(psf_norm)) * num_pupils_in_list


def test_get_normalization_all_zeros(make_fftpsf, set_test_backend):
    # Use small dimensions for these specific unit tests of _get_normalization
    # grid_size in make_fftpsf here will affect padding in the actual _get_normalization call
    fftpsf = make_fftpsf(num_rays=4, grid_size=4)

    pupil_all_zeros = be.zeros((4, 4), dtype=be.complex64)
    fftpsf.pupils = [pupil_all_zeros] # Override pupils

    # Test the actual _get_normalization method
    norm_factor = fftpsf._get_normalization()

    # Expected P_nom after be.where (which is what's FFT'd in _get_normalization)
    expected_P_nom_binary_for_fft = be.zeros_like(pupil_all_zeros)

    # The P_nom inside _get_normalization is padded before its FFT.
    # The helper needs to replicate this to get the correct expected value.
    # Let's get the padding amount used inside _get_normalization.
    # P_nom (binary mask) starts as self.pupils[0] shape.
    # It's then padded to grid_size for its internal FFT.
    # grid_size is fftpsf.grid_size, num_rays is fftpsf.num_rays (shape of pupils[0])
    pad_amount = (fftpsf.grid_size - fftpsf.num_rays) // 2
    if pad_amount > 0:
         expected_P_nom_binary_for_fft = be.pad(
            expected_P_nom_binary_for_fft,
            ((pad_amount, pad_amount), (pad_amount, pad_amount)),
            mode='constant',
            constant_values=0
        )

    expected_norm = calculate_expected_norm_for_test(expected_P_nom_binary_for_fft, len(fftpsf.pupils))
    assert_allclose(norm_factor, expected_norm, atol=1e-7)


def test_get_normalization_all_non_zeros(make_fftpsf, set_test_backend):
    fftpsf = make_fftpsf(num_rays=4, grid_size=4)

    pupil_all_non_zeros = be.ones((4, 4), dtype=be.complex64) # All 1s
    # Example with varied non-zero complex values
    # pupil_all_non_zeros = be.array([[0.5+0.1j, 1.0-0.2j], [1+1j, -0.3+0.8j]], dtype=be.complex64)
    # fftpsf = make_fftpsf(num_rays=pupil_all_non_zeros.shape[0], grid_size=pupil_all_non_zeros.shape[0])

    fftpsf.pupils = [pupil_all_non_zeros]

    norm_factor = fftpsf._get_normalization()

    expected_P_nom_binary_for_fft = be.ones_like(pupil_all_non_zeros)
    pad_amount = (fftpsf.grid_size - fftpsf.num_rays) // 2
    if pad_amount > 0:
        expected_P_nom_binary_for_fft = be.pad(
            expected_P_nom_binary_for_fft,
            ((pad_amount, pad_amount), (pad_amount, pad_amount)),
            mode='constant',
            constant_values=0
        )

    expected_norm = calculate_expected_norm_for_test(expected_P_nom_binary_for_fft, len(fftpsf.pupils))
    assert_allclose(norm_factor, expected_norm, atol=1e-7)


def test_get_normalization_mixed_zeros_non_zeros(make_fftpsf, set_test_backend):
    fftpsf = make_fftpsf(num_rays=4, grid_size=4)

    pupil_mixed_real_py = [
        [0.0, 1.0, 0.0, 2.0],
        [0.5, 0.0, 0.0, 0.8],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 3.0, 0.0]
    ]
    pupil_mixed_imag_py = [
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, -0.2],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0]
    ]
    pupil_mixed_real = be.array(pupil_mixed_real_py) # These are floats, should be fine
    pupil_mixed_imag = be.array(pupil_mixed_imag_py)
    # Construct complex array: C = R + i*I
    # Ensure parts are cast to complex before or during operation if necessary,
    # but backend should handle complex(float) + 1j * complex(float)
    # More safely: cast to complex64 first.
    pupil_mixed = be.astype(pupil_mixed_real, be.complex64) + 1j * be.astype(pupil_mixed_imag, be.complex64)

    fftpsf.pupils = [pupil_mixed]

    norm_factor = fftpsf._get_normalization()

    # Expected P_nom after be.where (binary mask based on pupil_mixed)
    expected_P_nom_binary_py = [
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [1, 0, 1, 0]
    ]
    # Create as backend array then cast. Helper `calculate_expected_norm_for_test` will cast to complex64.
    expected_P_nom_binary = be.astype(be.array(expected_P_nom_binary_py), be.float32)


    pad_amount = (fftpsf.grid_size - fftpsf.num_rays) // 2
    if pad_amount > 0:
        expected_P_nom_binary = be.pad(
            expected_P_nom_binary,
            ((pad_amount, pad_amount), (pad_amount, pad_amount)),
            mode='constant',
            constant_values=0
        )

    expected_norm = calculate_expected_norm_for_test(expected_P_nom_binary, len(fftpsf.pupils))
    assert_allclose(norm_factor, expected_norm, atol=1e-7)
