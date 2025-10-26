"""Unit tests for the ZernikePhaseProfile class."""

import pytest

from optiland import backend as be
from optiland.phase.zernike import ZernikePhaseProfile


@pytest.mark.parametrize("zernike_type", ["fringe", "noll", "standard"])
def test_zernike_phase_profile_init(zernike_type):
    """Test that ZernikePhaseProfile can be initialized with all zernike types."""
    zpp = ZernikePhaseProfile(zernike_type=zernike_type)
    assert zpp.zernike_type == zernike_type


def test_zernike_phase_profile_phase():
    """Test the get_phase method."""
    coeffs = be.array([0, 0, 0, 1])
    zpp = ZernikePhaseProfile(coefficients=coeffs, zernike_type="fringe", norm_radius=2.0)

    # Test at a known point
    x = be.array([1.0])
    y = be.array([0.0])
    phase = zpp.get_phase(x, y)
    # Z4 (defocus) is 2*rho^2 - 1. rho = 0.5. phase = 2*0.25 - 1 = -0.5
    assert be.isclose(phase, -0.5)


def test_zernike_phase_profile_gradient():
    """Test the get_gradient method."""
    coeffs = be.array([0, 0, 0, 1])
    zpp = ZernikePhaseProfile(coefficients=coeffs, zernike_type="fringe", norm_radius=2.0)

    # Test at a known point
    x = be.array([1.0])
    y = be.array([0.0])
    dx, dy = zpp.get_gradient(x, y)
    # dZ4/dx = d/dx(2*((x^2+y^2)/R^2) - 1) = 4x/R^2
    # dx = 4*1/4 = 1.0
    assert be.isclose(dx, 1.0)
    assert be.isclose(dy, 0.0)


def test_zernike_phase_profile_gradient_at_origin():
    """Test the get_gradient method at the origin."""
    coeffs = be.array([0, 0, 0, 1])
    zpp = ZernikePhaseProfile(coefficients=coeffs, zernike_type="fringe", norm_radius=2.0)

    # Test at the origin
    x = be.array([0.0])
    y = be.array([0.0])
    dx, dy = zpp.get_gradient(x, y)
    assert be.isclose(dx, 0.0)
    assert be.isclose(dy, 0.0)


@pytest.mark.parametrize("zernike_type", ["fringe", "noll", "standard"])
def test_zernike_phase_profile_to_from_dict(zernike_type):
    """Test the to_dict and from_dict methods."""
    coeffs = be.array([0, 1, 2, 3])
    zpp = ZernikePhaseProfile(coefficients=coeffs, zernike_type=zernike_type, norm_radius=3.0)
    zpp_dict = zpp.to_dict()

    zpp_from_dict = ZernikePhaseProfile.from_dict(zpp_dict)

    assert zpp_from_dict.zernike_type == zernike_type
    assert zpp_from_dict.norm_radius == 3.0
    assert be.allclose(zpp_from_dict.coefficients, coeffs)
