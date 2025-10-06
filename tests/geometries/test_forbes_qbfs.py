# tests/geometries/test_forbes_qbfs.py
"""
Tests for the ForbesQbfsGeometry class in optiland.geometries.
"""
import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.geometries import (
    ForbesQbfsGeometry,
    ForbesSurfaceConfig,
    ForbesSolverConfig,
    StandardGeometry,
)
from optiland.coordinate_system import CoordinateSystem
from optiland.materials.material import Material
from optiland.rays import RealRays
from ..utils import assert_allclose


def forbes_system():
    """
    Creates a standard optical system containing Forbes Q-bfs surfaces for
    testing.
    """
    lens = Optic()
    lens.set_aperture(aperture_type="EPD", value=4.0)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_wavelength(value=1.55, is_primary=True)
    H_K3 = Material("H-K3", reference="cdgm")
    H_ZLAF68C = Material("H-ZLAF68C", reference="cdgm")

    radial_terms_S2 = {0: 1.614, 1: 0.348, 2: 0.150, 3: 0.033, 4: 0.030}
    norm_radius_S2 = 6.336
    conic_S2 = -4.428

    radial_terms_S4 = {0: -0.270, 1: 0.087, 2: -0.048, 3: 0.026, 4: -0.012}
    conic_S4 = 0.038
    norm_radius_S4 = 10.0

    lens.add_surface(index=0, thickness=0.055)
    lens.add_surface(index=1, thickness=26.5)
    lens.add_surface(index=2, thickness=4.0, radius=be.inf, material=H_K3, is_stop=True)
    lens.add_surface(
        index=3,
        thickness=25.0,
        radius=22,
        conic=conic_S2,
        radial_terms=radial_terms_S2,
        norm_radius=norm_radius_S2,
        surface_type="forbes_qbfs",
    )
    lens.add_surface(index=4, thickness=7.0, radius=be.inf, material=H_ZLAF68C)
    lens.add_surface(
        index=5,
        thickness=10.0,
        radius=-31.0,
        conic=conic_S4,
        radial_terms=radial_terms_S4,
        norm_radius=norm_radius_S4,
        surface_type="forbes_qbfs",
    )
    lens.add_surface(index=6)
    return lens


class TestForbesQbfsGeometry:
    """
    Tests for the ForbesQbfsGeometry class, which represents a rotationally
    symmetric surface defined by Forbes Q-bfs polynomials.
    """

    def test_str(self, set_test_backend):
        """
        Tests the string representation of the geometry.
        """
        config = ForbesSurfaceConfig(radius=100.0)
        geometry = ForbesQbfsGeometry(CoordinateSystem(), surface_config=config)
        assert str(geometry) == "ForbesQbfs"

    def test_sag_with_infinite_radius(self, set_test_backend):
        """
        Tests sag calculation for a planar base surface. The sag should only
        consist of the aspheric departure.
        """
        config = ForbesSurfaceConfig(radius=be.inf, terms={1: 1e-3}, norm_radius=10.0)
        geometry = ForbesQbfsGeometry(CoordinateSystem(), surface_config=config)
        x, y = 5.0, 0.0
        rho = 5.0
        u = rho / 10.0
        usq = u**2
        q1 = (13 - 16 * usq) / be.sqrt(be.array(19.0))
        expected_sag = usq * (1 - usq) * 1e-3 * q1
        assert_allclose(geometry.sag(x, y), expected_sag)

    def test_sag_outside_norm_radius(self, set_test_backend):
        """
        Tests that the aspheric departure is zero outside the normalization
        radius, so the sag equals the base conic sag.
        """
        config = ForbesSurfaceConfig(radius=100.0, terms={0: 1e-3}, norm_radius=10.0)
        geometry = ForbesQbfsGeometry(CoordinateSystem(), surface_config=config)
        standard_geom = StandardGeometry(CoordinateSystem(), radius=100.0)
        x, y = 12.0, 0.0
        assert_allclose(geometry.sag(x, y), standard_geom.sag(x, y))

    def test_analytical_normal_vs_autodiff(self, set_test_backend):
        """
        Compares the analytical surface normal with a numerical gradient from
        autodiff for validation.
        """
        if be.get_backend() != "torch":
            pytest.skip("This test requires the torch backend.")

        radial_terms = {0: 1.6e-4, 1: 0.3e-4, 2: 0.15e-4}
        config = ForbesSurfaceConfig(
            radius=22.0, conic=-4.428, terms=radial_terms, norm_radius=6.336
        )
        geometry_torch = ForbesQbfsGeometry(CoordinateSystem(), surface_config=config)
        x, y = be.tensor(2.5, requires_grad=True), be.tensor(1.5, requires_grad=True)

        sag = geometry_torch.sag(x, y)
        sag.backward()

        nx_autodiff, ny_autodiff = -x.grad, -y.grad
        nz_autodiff = be.sqrt(1 - (nx_autodiff**2 + ny_autodiff**2))

        nx_analytical, ny_analytical, _ = geometry_torch._surface_normal(be.to_numpy(x), be.to_numpy(y))

        assert_allclose(be.to_numpy(nx_autodiff), nx_analytical, atol=1e-7)
        assert_allclose(be.to_numpy(ny_autodiff), ny_analytical, atol=1e-7)

    def test_sag_vs_zemax(self, set_test_backend):
        """
        Compares the sag calculation against reference values from Zemax.
        """
        radial_terms = {n: c for n, c in enumerate([1.614, 0.348, 0.150, 0.033, 0.030])}
        config = ForbesSurfaceConfig(
            radius=21.723, conic=-4.428, terms=radial_terms, norm_radius=6.336
        )
        geometry = ForbesQbfsGeometry(CoordinateSystem(), surface_config=config)
        y_coords = be.array([0.75, 1.0, 1.250, 1.860])
        zemax_sag_values = be.array([6.26e-02, 1.08e-01, 1.64e-01, 3.30e-01])
        optiland_sag = geometry.sag(y=y_coords, x=0)
        assert be.allclose(optiland_sag, zemax_sag_values, atol=1e-2)

    def test_surface_normal_at_vertex(self, set_test_backend):
        """
        Tests that the surface normal at the vertex is always [0, 0, -1].
        """
        config = ForbesSurfaceConfig(
            radius=50.0, conic=-1.0, terms={2: 1e-4}, norm_radius=10.0
        )
        geometry = ForbesQbfsGeometry(CoordinateSystem(), surface_config=config)
        nx, ny, nz = geometry._surface_normal(x=0.0, y=0.0)
        assert_allclose(nx, 0.0)
        assert_allclose(ny, 0.0)
        assert_allclose(nz, -1.0)

    def test_tracing(self, set_test_backend):
        """
        Tests ray tracing through a full optical system containing Q-bfs
        surfaces and compares against reference Zemax data.
        """
        system = forbes_system()
        test_rays = RealRays(
            x=0, y=[-2.0, 0.0, 2.0], z=0, L=0, M=0, N=1, wavelength=1.550
        )
        zmx_rays_y = be.array([-5.863, 0.0, 5.863])
        rays_out = system.surface_group.trace(test_rays)
        assert be.allclose(rays_out.y, zmx_rays_y, atol=1e-3)

    def test_to_dict_from_dict(self, set_test_backend):
        """
        Tests the serialization to and deserialization from a dictionary.
        """
        cs = CoordinateSystem(x=1, y=-1, z=10, rx=0.01, ry=0.02, rz=-0.03)
        radial_terms = {0: 1e-3, 1: 2e-4, 2: -5e-5}
        surface_config = ForbesSurfaceConfig(
            radius=123.4, conic=-0.9, terms=radial_terms, norm_radius=45.6
        )
        solver_config = ForbesSolverConfig(tol=1e-12, max_iter=75)
        original = ForbesQbfsGeometry(cs, surface_config, solver_config)

        geom_dict = original.to_dict()
        reconstructed = ForbesQbfsGeometry.from_dict(geom_dict)

        assert reconstructed.to_dict() == geom_dict