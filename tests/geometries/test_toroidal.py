# tests/geometries/test_toroidal.py
"""
Tests for the ToroidalGeometry class in optiland.geometries.
"""
import pytest

import optiland.backend as be
from optiland import geometries
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays import RealRays
from ..utils import assert_allclose


@pytest.fixture
def basic_toroid_geometry(set_test_backend):
    """
    Provides a basic ToroidalGeometry instance for testing.
    """
    cs = CoordinateSystem(x=0, y=0, z=0)
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_x=100.0,
        radius_y=50.0,
        conic=-0.5,
        coeffs_poly_y=[1e-5],
    )


@pytest.fixture
def cylinder_x_geometry(set_test_backend):
    """
    Provides a cylindrical geometry that is flat in the X-Z plane (Rx = inf).
    """
    cs = CoordinateSystem(x=0, y=0, z=0)
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_x=be.inf,
        radius_y=-50.0,
        conic=0.0,
    )


@pytest.fixture
def cylinder_y_geometry(set_test_backend):
    """
    Provides a cylindrical geometry that is flat in the Y-Z plane (Ry = inf).
    """
    cs = CoordinateSystem(x=0, y=0, z=0)
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_x=100.0,
        radius_y=be.inf,
        conic=0.0,
    )


class TestToroidalGeometry:
    """
    Tests for the ToroidalGeometry class, which represents a surface with
    different curvatures in the x and y directions.
    """

    def test_toroidal_str(self, set_test_backend):
        """
        Tests the string representation of the ToroidalGeometry.
        """
        cs = CoordinateSystem()
        geometry = geometries.ToroidalGeometry(
            cs,
            radius_x=100.0,
            radius_y=50.0,
            conic=-0.5,
            coeffs_poly_y=[1e-5],
        )
        assert str(geometry) == "Toroidal"

    def test_toroidal_sag_vertex(self, basic_toroid_geometry, set_test_backend):
        """
        Tests that the sag at the vertex (0, 0) is zero.
        """
        assert be.allclose(basic_toroid_geometry.sag(0.0, 0.0), 0.0)

    def test_toroidal_normal_vertex(self, basic_toroid_geometry, set_test_backend):
        """
        Tests that the normal vector at the vertex is [0, 0, -1].
        """
        nx, ny, nz = basic_toroid_geometry._surface_normal(0.0, 0.0)
        assert be.allclose(nx, 0.0)
        assert be.allclose(ny, 0.0)
        assert be.allclose(nz, -1.0)

    def test_toroidal_sag_known_points(self, basic_toroid_geometry, set_test_backend):
        """
        Tests the sag calculation at several known off-axis points.
        """
        x = be.array([0.0, 10.0, 0.0])
        y = be.array([10.0, 0.0, 5.0])
        expected_z = be.array([1.00605051, 0.50125628, 0.25056330])
        calculated_z = basic_toroid_geometry.sag(x, y)
        assert be.allclose(calculated_z, expected_z)

    def test_toroidal_normal_known_points(
        self, basic_toroid_geometry, set_test_backend
    ):
        """
        Tests the surface normal calculation at several known off-axis points.
        """
        x = be.array([0.0, 10.0])
        y = be.array([10.0, 0.0])
        expected_nx = be.array([0.0, 0.10000])
        expected_ny = be.array([0.198219, 0.0])
        expected_nz = be.array([-0.980158, -0.994987])
        nx, ny, nz = basic_toroid_geometry._surface_normal(x, y)
        assert be.allclose(nx, expected_nx)
        assert be.allclose(ny, expected_ny)
        assert be.allclose(nz, expected_nz)

    def test_cylinder_x_sag(self, cylinder_x_geometry, set_test_backend):
        """
        Tests the sag for a cylinder flat in X. The sag should only depend on y.
        """
        x = be.array([0.0, 10.0, 10.0])
        y = be.array([5.0, 5.0, 0.0])
        expected_z = be.array([-0.25062818, -0.25062818, 0.0])
        calculated_z = cylinder_x_geometry.sag(x, y)
        assert_allclose(calculated_z, expected_z)

    def test_cylinder_y_sag(self, cylinder_y_geometry, set_test_backend):
        """
        Tests the sag for a cylinder flat in Y. The sag should only depend on x.
        """
        x = be.array([5.0, 5.0, 0.0])
        y = be.array([0.0, 10.0, 10.0])
        expected_z = be.array([0.12507822, 0.12507822, 0.0])
        calculated_z = cylinder_y_geometry.sag(x, y)
        assert_allclose(calculated_z, expected_z)

    def test_toroidal_sag_vs_zemax(self, basic_toroid_geometry, set_test_backend):
        """
        Compares sag values calculated by Optiland with reference Zemax data.
        """
        geometry = basic_toroid_geometry
        x = be.array([0.0, 2.5, 0.0, -2.5, 5.0, -5.0, 2.5, -2.5])
        y = be.array([0.0, 0.0, 2.5, 0.0, 2.5, -2.5, -2.5, 2.5])
        zemax_z_sag = be.array(
            [
                0.0,
                3.12548843e-02,
                6.25820434e-02,
                3.12548843e-02,
                1.87738689e-01,
                1.87738689e-01,
                9.38565061e-02,
                9.38565061e-02,
            ]
        )
        optiland_z_sag = geometry.sag(x, y)
        assert be.allclose(optiland_z_sag, zemax_z_sag)

    def test_toroidal_ray_tracing_comparison(self, set_test_backend):
        """
        Traces rays through a single toroidal surface and compares the output
        with reference Zemax ray tracing data.
        """
        lens = Optic()
        lens.add_surface(index=0, thickness=be.inf)
        lens.add_surface(
            index=1,
            surface_type="toroidal",
            thickness=5.0,
            material=IdealMaterial(n=1.5),
            is_stop=True,
            radius_x=100.0,
            radius_y=50.0,
            conic=-0.5,
            toroidal_coeffs_poly_y=[0.05, 0.0002],
        )
        lens.add_surface(index=2, thickness=10.0, material="air")
        lens.add_surface(index=3)
        lens.set_aperture(aperture_type="EPD", value=10.0)
        lens.add_wavelength(value=0.550, is_primary=True)
        lens.set_field_type("angle")
        lens.add_field(y=0)

        # Tangential (Y) Fan Test
        y_coords = be.linspace(-5.0, 5.0, 5)
        rays_in_yfan = RealRays(x=0, y=y_coords, z=0, L=0, M=0, N=1, wavelength=0.55)
        rays_out_yfan = lens.surface_group.trace(rays_in_yfan)
        zemax_y_out_yfan = be.array([-0.8123, -0.4676, 0, 0.4676, 0.8123])
        assert be.allclose(rays_out_yfan.y, zemax_y_out_yfan, atol=1e-4)

        # Sagittal (X) Fan Test
        x_coords = be.linspace(-5.0, 5.0, 5)
        rays_in_xfan = RealRays(x=x_coords, y=0, z=0, L=0, M=0, N=1, wavelength=0.55)
        rays_out_xfan = lens.surface_group.trace(rays_in_xfan)
        zemax_x_out_xfan = be.array([-4.6683, -2.3335, 0, 2.3335, 4.6683])
        assert be.allclose(rays_out_xfan.x, zemax_x_out_xfan, atol=1e-4)

    def test_toroidal_to_dict(self, basic_toroid_geometry, set_test_backend):
        """
        Tests serialization of a ToroidalGeometry instance to a dictionary.
        """
        geom_dict = basic_toroid_geometry.to_dict()
        assert geom_dict["type"] == "ToroidalGeometry"
        assert geom_dict["radius_x"] == 100.0
        assert geom_dict["radius_y"] == 50.0
        assert geom_dict["conic_yz"] == -0.5
        assert geom_dict["coeffs_poly_y"] == pytest.approx([1e-5])

    def test_toroidal_from_dict(self, basic_toroid_geometry, set_test_backend):
        """
        Tests deserialization of a ToroidalGeometry instance from a dictionary.
        """
        geom_dict = basic_toroid_geometry.to_dict()
        new_geometry = geometries.ToroidalGeometry.from_dict(geom_dict)
        assert isinstance(new_geometry, geometries.ToroidalGeometry)
        assert new_geometry.to_dict() == geom_dict

    def test_toroidal_from_dict_invalid(self, set_test_backend):
        """
        Tests that deserializing from a dictionary with missing required keys
        raises a ValueError.
        """
        cs = CoordinateSystem()
        invalid_dict = {"type": "ToroidalGeometry", "cs": cs.to_dict()}
        with pytest.raises(ValueError):
            geometries.ToroidalGeometry.from_dict(invalid_dict)