# tests/geometries/test_forbes_q2d.py
"""
Tests for the ForbesQ2dGeometry class in optiland.geometries.
"""
import pytest
import numpy as np

import optiland.backend as be
from optiland.optic import Optic
from optiland.geometries import (
    ForbesQ2dGeometry,
    ForbesQbfsGeometry,
    ForbesSurfaceConfig,
    StandardGeometry,
)
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.rays import RealRays
from ..utils import assert_allclose


class TestForbesQ2dGeometry:
    """
    Tests for the ForbesQ2dGeometry class, which represents a freeform surface
    defined by Forbes Q2D polynomials.
    """

    def test_str(self, set_test_backend):
        """
        Tests the string representation of the geometry.
        """
        config = ForbesSurfaceConfig(radius=100.0, conic=0.0)
        geometry = ForbesQ2dGeometry(CoordinateSystem(), surface_config=config)
        assert str(geometry) == "ForbesQ2d"

    def test_init_no_coeffs(self, set_test_backend):
        """
        Tests initialization with no freeform coefficients. The coefficient
        lists should be empty.
        """
        config = ForbesSurfaceConfig(radius=100.0, conic=-0.5)
        geometry = ForbesQ2dGeometry(CoordinateSystem(), surface_config=config)
        assert len(geometry.coeffs_c) == 0
        assert len(geometry.coeffs_n) == 0
        assert len(geometry.cm0_coeffs) == 0
        assert len(geometry.ams_coeffs) == 0
        assert len(geometry.bms_coeffs) == 0

    def test_sag_symmetric_terms_only(self, set_test_backend):
        """
        Tests that with only symmetric (m=0) terms, the sag matches a Q-bfs
        surface.
        """
        radial_terms = {0: 1e-3, 1: -2e-4}
        freeform_coeffs = {("a", 0, n): c for n, c in radial_terms.items()}
        q2d_config = ForbesSurfaceConfig(
            radius=50.0, conic=0.0, terms=freeform_coeffs, norm_radius=10.0
        )
        geom_q2d = ForbesQ2dGeometry(CoordinateSystem(), surface_config=q2d_config)
        qbfs_config = ForbesSurfaceConfig(
            radius=50.0, conic=0.0, terms=radial_terms, norm_radius=10.0
        )
        geom_qbfs = ForbesQbfsGeometry(CoordinateSystem(), surface_config=qbfs_config)
        x, y = 3.0, 4.0
        assert_allclose(geom_q2d.sag(x, y), geom_qbfs.sag(x, y))

    def test_sag_with_sine_term(self, set_test_backend):
        """
        Tests that a sine term (b_m^n) produces zero departure along the
        x-axis (where sin(m*theta) is zero).
        """
        config = ForbesSurfaceConfig(
            radius=100.0, conic=0.0, terms={("b", 1, 1): 1e-3}, norm_radius=10.0
        )
        geometry = ForbesQ2dGeometry(CoordinateSystem(), surface_config=config)
        x, y = 5.0, 0.0
        base_geom = StandardGeometry(CoordinateSystem(), radius=100.0)
        assert_allclose(geometry.sag(x, y), base_geom.sag(x, y))

    def test_prepare_coeffs(self, set_test_backend):
        """
        Tests the internal _prepare_coeffs method to ensure it correctly
        parses the dictionary of coefficients into structured lists.
        """
        freeform_coeffs = {
            ("a", 0, 0): 1.0,
            ("a", 1, 1): 2.0,
            ("b", 1, 1): 3.0,
            ("a", 0, 4): 4.0,
        }
        config = ForbesSurfaceConfig(
            radius=100.0, conic=0.0, terms=freeform_coeffs, norm_radius=10.0
        )
        geometry = ForbesQ2dGeometry(CoordinateSystem(), surface_config=config)
        assert_allclose(geometry.cm0_coeffs, [1.0, 0.0, 0.0, 0.0, 4.0])
        assert len(geometry.ams_coeffs) == 1
        assert_allclose(geometry.ams_coeffs[0], [0.0, 2.0])
        assert len(geometry.bms_coeffs) == 1
        assert_allclose(geometry.bms_coeffs[0], [0.0, 3.0])

    def test_to_dict_from_dict(self, set_test_backend):
        """
        Tests the serialization to and deserialization from a dictionary.
        """
        cs = CoordinateSystem(x=1, y=-1, z=10)
        freeform_coeffs = {("a", 2, 2): 1e-4, ("b", 1, 1): -5e-5}
        config = ForbesSurfaceConfig(
            radius=123.4, conic=-0.9, terms=freeform_coeffs, norm_radius=45.6
        )
        original = ForbesQ2dGeometry(CoordinateSystem(), surface_config=config)
        geom_dict = original.to_dict()
        reconstructed = ForbesQ2dGeometry.from_dict(geom_dict)
        assert reconstructed.to_dict() == geom_dict


class TestForbesValidation:
    """
    Unit tests to validate the Forbes geometry implementation directly against
    the mathematical formulas in the reference papers.
    """

    def test_qnm_values_against_analytical_formula(self, set_test_backend):
        """
        Validates that a single Q polynomial from the qpoly implementation
        matches a value calculated directly from the analytical formula.
        """
        n, m = 1, 2
        x = 0.4
        P_n_m_canonical = 1.5 - x
        from optiland.geometries.forbes.qpoly import f_q2d, g_q2d

        P_0_m = 0.5
        f0 = f_q2d(n=0, m=m)
        Q_0_m_canonical = P_0_m / f0
        g0 = g_q2d(n=0, m=m)
        f1 = f_q2d(n=1, m=m)
        Q_n_m_canonical = (P_n_m_canonical - g0 * Q_0_m_canonical) / f1
        coeffs_to_test = [0.0] * (n + 1)
        coeffs_to_test[n] = 1.0
        from optiland.geometries.forbes.qpoly import clenshaw_q2d

        alphas = clenshaw_q2d(coeffs_to_test, m=m, usq=x)
        Q_n_m_optiland = 0.5 * alphas[0]
        assert be.allclose(Q_n_m_optiland, Q_n_m_canonical, atol=1e-9)

    def test_q2d_normal_against_numerical_derivative(self):
        """
        Validates the analytical surface normal against a numerical derivative
        (finite difference).
        """
        freeform_coeffs = {("a", 1, 1): -0.25, ("b", 1, 0): 0.5}
        config = ForbesSurfaceConfig(
            radius=21.709, conic=-4.428, terms=freeform_coeffs, norm_radius=6.0
        )
        geometry = ForbesQ2dGeometry(CoordinateSystem(), surface_config=config)
        x, y = 1.0, 0.5
        h = 1e-6
        sag_x_plus = geometry.sag(x + h, y)
        sag_x_minus = geometry.sag(x - h, y)
        sag_y_plus = geometry.sag(x, y + h)
        sag_y_minus = geometry.sag(x, y - h)
        df_dx_numerical = (sag_x_plus - sag_x_minus) / (2 * h)
        df_dy_numerical = (sag_y_plus - sag_y_minus) / (2 * h)
        df_dx_analytical, df_dy_analytical = geometry._surface_normal_analytical(
            be.array(x), be.array(y)
        )
        assert np.allclose(df_dx_analytical, df_dx_numerical, atol=1e-6)
        assert np.allclose(df_dy_analytical, df_dy_numerical, atol=1e-6)

    def test_complex_ray_tracing(self, set_test_backend):
        """
        Tests ray tracing through a complex system with both Q-2d and Q-bfs
        surfaces and compares against reference Zemax data.
        """
        optic = Optic()
        optic.set_aperture(aperture_type="EPD", value=4.0)
        optic.set_field_type(field_type="angle")
        optic.add_field(y=0)
        optic.add_wavelength(value=1.550, is_primary=True)
        H_K3 = IdealMaterial(n=1.50)
        H_ZLAF68C = Material("H-ZLAF68C", reference="cdgm")
        freeform_coeffs = {
            ("b", 1, 0): 0.23, ("a", 1, 1): -0.25, ("a", 1, 3): -2.0,
            ("b", 2, 0): 0.4, ("b", 3, 1): 0.5,
        }
        radial_terms = {0: -0.334, 1: 0.130, 2: -0.099, 3: 0.082, 4: -0.093}
        optic.add_surface(index=0, thickness=be.inf)
        optic.add_surface(index=1, thickness=26.5)
        optic.add_surface(index=2, thickness=4.0, radius=be.inf, material=H_K3, is_stop=True)
        optic.add_surface(index=3, thickness=25.0, radius=21.7, conic=-4.428, freeform_coeffs=freeform_coeffs, norm_radius=6.0, surface_type="forbes_q2d")
        optic.add_surface(index=4, thickness=7.0, radius=be.inf, material=H_ZLAF68C)
        optic.add_surface(index=5, thickness=10.0, radius=-31.408, conic=-0.334, radial_terms=radial_terms, norm_radius=10.0, surface_type="forbes_qbfs")
        optic.add_surface(index=6)

        rays_in = RealRays(x=[-2.0, 0.0, 2.0], y=0, z=0, L=0, M=0, N=1, wavelength=1.550)
        rays_out = optic.surface_group.trace(rays_in)

        rays_zmx = be.array([[2.262, -7.611, 8.507], [0.701, 0.635, 1.844]])
        assert be.allclose(rays_out.x, rays_zmx[0], atol=1e-3)
        assert be.allclose(rays_out.y, rays_zmx[1], atol=1e-3)