# tests/geometries/test_flip_geometries.py
"""
Tests the `flip()` method for all geometry classes in optiland.geometries.

The `flip()` method is essential for reversing optical systems. These tests
ensure that each geometry type correctly modifies its parameters when flipped,
which typically involves negating the radius of curvature.
"""
import pytest
import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import (
    StandardGeometry,
    Plane,
    BiconicGeometry,
    ChebyshevPolynomialGeometry,
    EvenAsphere,
    OddAsphere,
    PolynomialGeometry,
    ToroidalGeometry,
    ZernikePolynomialGeometry,
)

# A default, non-tilted, centered coordinate system for all tests.
cs = CoordinateSystem()


def test_flip_standard_geometry():
    """
    Tests flipping a StandardGeometry. Expects the radius to be negated and
    the conic constant to remain unchanged.
    """
    initial_radius = 10.0
    initial_conic = -0.5
    geom = StandardGeometry(cs, radius=initial_radius, conic=initial_conic)
    geom.flip()
    assert geom.radius == -initial_radius
    assert geom.k == initial_conic


def test_flip_plane_geometry():
    """
    Tests flipping a Plane geometry. Expects the radius (infinity) to
    remain unchanged.
    """
    geom = Plane(cs)
    initial_radius = geom.radius
    geom.flip()
    assert geom.radius == initial_radius


def test_flip_biconic_geometry():
    """
    Tests flipping a BiconicGeometry. Expects both x and y radii to be
    negated, while conic constants remain unchanged.
    """
    initial_rx, initial_ry = 10.0, -20.0
    initial_kx, initial_ky = 0.1, -0.2
    geom = BiconicGeometry(
        cs,
        radius_x=initial_rx,
        radius_y=initial_ry,
        conic_x=initial_kx,
        conic_y=initial_ky,
    )
    geom.flip()
    assert geom.Rx == -initial_rx
    assert geom.Ry == -initial_ry
    assert geom.kx == initial_kx
    assert geom.ky == initial_ky


def test_flip_chebyshev_geometry():
    """
    Tests flipping a ChebyshevPolynomialGeometry. Expects the base radius to be
    negated, while the conic and coefficients remain unchanged.
    """
    initial_radius = 100.0
    initial_coeffs = [[0, 0.1], [0.2, 0.3]]
    geom = ChebyshevPolynomialGeometry(
        cs,
        radius=initial_radius,
        conic=-0.8,
        coefficients=initial_coeffs,
        norm_x=1.0,
        norm_y=1.0,
    )
    geom.flip()
    assert geom.radius == -initial_radius
    assert be.allclose(geom.coefficients, be.array(initial_coeffs))


def test_flip_even_asphere_geometry():
    """
    Tests flipping an EvenAsphere geometry. Expects the base radius to be
    negated, while the conic and coefficients remain unchanged.
    """
    initial_radius = 50.0
    initial_coeffs = [0.001, -0.0005]
    geom = EvenAsphere(cs, radius=initial_radius, conic=0.0, coefficients=initial_coeffs)
    geom.flip()
    assert geom.radius == -initial_radius
    assert geom.coefficients == initial_coeffs


def test_flip_odd_asphere_geometry():
    """
    Tests flipping an OddAsphere geometry. Expects the base radius to be
    negated, while the conic and coefficients remain unchanged.
    """
    initial_radius = 75.0
    initial_coeffs = [0.01, 0.002]
    geom = OddAsphere(
        cs, radius=initial_radius, conic=-1.0, coefficients=initial_coeffs
    )
    geom.flip()
    assert geom.radius == -initial_radius
    assert geom.coefficients == initial_coeffs


def test_flip_polynomial_geometry():
    """
    Tests flipping a PolynomialGeometry. Expects the base radius to be
    negated, while the conic and coefficients remain unchanged.
    """
    initial_radius = 120.0
    initial_coeffs = [[0, 0.01, 0.02], [0.005, 0, 0]]
    geom = PolynomialGeometry(
        cs, radius=initial_radius, conic=0.5, coefficients=initial_coeffs
    )
    geom.flip()
    assert geom.radius == -initial_radius
    assert be.allclose(geom.coefficients, be.array(initial_coeffs))


def test_flip_toroidal_geometry():
    """
    Tests flipping a ToroidalGeometry. Expects both rotational and YZ radii
    to be negated, while other parameters remain unchanged.
    """
    initial_r_rot, initial_r_yz = 200.0, -100.0
    geom = ToroidalGeometry(
        cs, radius_x=initial_r_rot, radius_y=initial_r_yz, conic=-0.7
    )
    geom.flip()
    assert geom.R_rot == -initial_r_rot
    assert geom.R_yz == -initial_r_yz


def test_flip_zernike_geometry():
    """
    Tests flipping a ZernikePolynomialGeometry. Expects the base radius to be
    negated, while the conic and coefficients remain unchanged.
    """
    initial_radius = 300.0
    initial_coeffs = [0, 0, 0.05, -0.03]
    geom = ZernikePolynomialGeometry(
        cs, radius=initial_radius, conic=-0.2, coefficients=initial_coeffs, norm_radius=10.0
    )
    geom.flip()
    assert geom.radius == -initial_radius
    assert be.allclose(geom.coefficients, be.array(initial_coeffs))


def test_flip_biconic_zero_radius():
    """
    Tests flipping a BiconicGeometry where one radius is zero. The zero
    radius should remain zero.
    """
    geom = BiconicGeometry(cs, radius_x=0.0, radius_y=-20.0)
    geom.flip()
    assert geom.Rx == 0.0
    assert geom.Ry == 20.0


def test_flip_toroidal_zero_radius():
    """
    Tests flipping a ToroidalGeometry where one radius is zero. The zero
    radius should remain zero.
    """
    geom = ToroidalGeometry(cs, radius_x=0.0, radius_y=-100.0)
    geom.flip()
    assert geom.R_rot == 0.0
    assert geom.R_yz == 100.0


def test_flip_standard_inf_radius():
    """
    Tests flipping a StandardGeometry with an infinite radius. The radius
    should be negated to -inf.
    """
    geom = StandardGeometry(cs, radius=be.inf)
    geom.flip()
    assert geom.radius == -be.inf


def test_flip_biconic_inf_radius():
    """
    Tests flipping a BiconicGeometry where one radius is infinite. The
    infinite radius should be negated.
    """
    geom = BiconicGeometry(cs, radius_x=be.inf, radius_y=-20.0)
    geom.flip()
    assert geom.Rx == -be.inf
    assert geom.Ry == 20.0


def test_flip_toroidal_inf_radius():
    """
    Tests flipping a ToroidalGeometry where one radius is infinite. The
    infinite radius should be negated.
    """
    geom = ToroidalGeometry(cs, radius_x=be.inf, radius_y=-100.0)
    geom.flip()
    assert geom.R_rot == -be.inf
    assert geom.R_yz == 100.0