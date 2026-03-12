"""Tests for VectorialHuygensPSF and HuygensPSF factory dispatch."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import optiland.backend as be
from optiland.psf.huygens_fresnel import HuygensPSF, ScalarHuygensPSF
from optiland.psf.vectorial_huygens import VectorialHuygensPSF
from optiland.rays import PolarizationState

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def set_numpy_backend():
    """Force the numpy backend for all tests and restore afterwards."""
    original = be.get_backend()
    be.set_backend("numpy")
    yield
    be.set_backend(original)


@pytest.fixture
def cooke_triplet():
    from optiland.samples.objectives import CookeTriplet

    return CookeTriplet()


@pytest.fixture
def reverse_telephoto():
    from optiland.samples.objectives import ReverseTelephoto

    return ReverseTelephoto()


@pytest.fixture
def polarized_cooke_triplet(cooke_triplet):
    state = PolarizationState(
        is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0
    )
    cooke_triplet.set_polarization(state)
    return cooke_triplet


@pytest.fixture
def unpolarized_cooke_triplet(cooke_triplet):
    state = PolarizationState(is_polarized=False)
    cooke_triplet.set_polarization(state)
    return cooke_triplet


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_factory_returns_scalar_when_no_polarization(cooke_triplet):
    """HuygensPSF factory should return ScalarHuygensPSF with no polarization."""
    psf = HuygensPSF(
        cooke_triplet, field=(0.0, 0.0), wavelength="primary",
        num_rays=32, image_size=32,
    )
    assert isinstance(psf, ScalarHuygensPSF)
    assert not isinstance(psf, VectorialHuygensPSF)


def test_factory_returns_vectorial_when_polarized(polarized_cooke_triplet):
    """HuygensPSF factory should return VectorialHuygensPSF when polarization is set."""
    psf = HuygensPSF(
        polarized_cooke_triplet, field=(0.0, 0.0), wavelength="primary",
        num_rays=32, image_size=32,
    )
    assert isinstance(psf, VectorialHuygensPSF)


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


def test_vectorial_psf_shape(polarized_cooke_triplet):
    """PSF array should have shape (image_size, image_size)."""
    image_size = 32
    psf = VectorialHuygensPSF(
        polarized_cooke_triplet, field=(0.0, 0.0), wavelength="primary",
        num_rays=32, image_size=image_size,
    )
    assert psf.psf.shape == (image_size, image_size)


def test_vectorial_psf_non_negative(polarized_cooke_triplet):
    """PSF values must be non-negative (intensities)."""
    psf = VectorialHuygensPSF(
        polarized_cooke_triplet, field=(0.0, 0.0), wavelength="primary",
        num_rays=32, image_size=32,
    )
    assert np.all(be.to_numpy(psf.psf) >= 0)


def test_vectorial_psf_normalization_positive(polarized_cooke_triplet):
    """Normalization factor must be strictly positive."""
    psf = VectorialHuygensPSF(
        polarized_cooke_triplet, field=(0.0, 0.0), wavelength="primary",
        num_rays=32, image_size=32,
    )
    norm = psf._get_normalization()
    assert float(be.to_numpy(np.array(norm))) > 0


# ---------------------------------------------------------------------------
# Strehl ratio
# ---------------------------------------------------------------------------


def test_vectorial_strehl_positive(polarized_cooke_triplet):
    """Vectorial Strehl ratio must be a positive finite value <= 1.0."""
    psf = VectorialHuygensPSF(
        polarized_cooke_triplet, field=(0.0, 0.0), wavelength="primary",
        num_rays=64, image_size=64,
    )
    strehl = psf.strehl_ratio()
    assert 0.0 < strehl <= 1.0


def test_vectorial_strehl_matches_scalar_on_axis(cooke_triplet):
    """Vectorial and scalar Strehl should agree within 1% for an uncoated lens."""
    psf_scalar = ScalarHuygensPSF(
        cooke_triplet, field=(0.0, 0.0), wavelength="primary",
        num_rays=64, image_size=64,
    )

    state = PolarizationState(
        is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0
    )
    cooke_triplet.set_polarization(state)
    psf_vectorial = VectorialHuygensPSF(
        cooke_triplet, field=(0.0, 0.0), wavelength="primary",
        num_rays=64, image_size=64,
    )

    assert abs(psf_vectorial.strehl_ratio() - psf_scalar.strehl_ratio()) < 0.01


# ---------------------------------------------------------------------------
# Off-axis field
# ---------------------------------------------------------------------------


def test_vectorial_off_axis_psf_valid(polarized_cooke_triplet):
    """VectorialHuygensPSF should compute a valid PSF for an off-axis field."""
    psf = VectorialHuygensPSF(
        polarized_cooke_triplet, field=(0.0, 0.7), wavelength="primary",
        num_rays=32, image_size=32,
    )
    arr = be.to_numpy(psf.psf)
    assert arr.shape == (32, 32)
    assert np.all(arr >= 0)
    assert arr.max() > 0


# ---------------------------------------------------------------------------
# Unpolarized (two incoherent states)
# ---------------------------------------------------------------------------


def test_factory_with_unpolarized_state_returns_vectorial(unpolarized_cooke_triplet):
    """Unpolarized PolarizationState (is_polarized=False) should still dispatch
    to VectorialHuygensPSF via the factory."""
    psf = HuygensPSF(
        unpolarized_cooke_triplet, field=(0.0, 0.0), wavelength="primary",
        num_rays=32, image_size=32,
    )
    assert isinstance(psf, VectorialHuygensPSF)


def test_vectorial_unpolarized_psf_valid(unpolarized_cooke_triplet):
    """An unpolarized source should produce a non-negative PSF with positive peak."""
    psf = VectorialHuygensPSF(
        unpolarized_cooke_triplet, field=(0.0, 0.0), wavelength="primary",
        num_rays=32, image_size=32,
    )
    arr = be.to_numpy(psf.psf)
    assert np.all(arr >= 0)
    assert arr.max() > 0


# ---------------------------------------------------------------------------
# View method
# ---------------------------------------------------------------------------


def test_vectorial_view_runs(polarized_cooke_triplet):
    """view() should run without error for the vectorial Huygens PSF."""
    psf = VectorialHuygensPSF(
        polarized_cooke_triplet, field=(0.0, 0.0), wavelength="primary",
        num_rays=32, image_size=32,
    )
    fig, _ = psf.view()
    plt.close(fig)


# ---------------------------------------------------------------------------
# pixel_pitch and oversample pass-through
# ---------------------------------------------------------------------------


def test_vectorial_pixel_pitch_override(polarized_cooke_triplet):
    """pixel_pitch kwarg should be forwarded correctly."""
    psf = VectorialHuygensPSF(
        polarized_cooke_triplet, field=(0.0, 0.0), wavelength="primary",
        num_rays=32, image_size=32, pixel_pitch=0.005,
    )
    assert psf.pixel_pitch == pytest.approx(0.005)
    assert psf.psf.shape == (32, 32)
