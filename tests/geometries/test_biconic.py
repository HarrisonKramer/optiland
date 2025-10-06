# tests/geometries/test_biconic.py
"""
Tests for the BiconicGeometry class in optiland.geometries.
"""
import pytest

import optiland.backend as be
from optiland.geometries import BiconicGeometry
from optiland.coordinate_system import CoordinateSystem
from optiland.rays import RealRays
from ..utils import assert_allclose


class TestBiconicGeometry:
    """
    Tests for the BiconicGeometry class, which represents a surface with
    different conic profiles in the x and y directions.
    """

    def test_str(self, set_test_backend):
        """
        Tests the string representation of the BiconicGeometry.
        """
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=10.0, radius_y=20.0)
        assert str(geom) == "Biconic"

    def test_sag_vertex(self, set_test_backend):
        """
        Tests that the sag at the vertex (0, 0) is always zero.
        """
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=20.0, conic_x=0.5, conic_y=-0.5
        )
        assert_allclose(geom.sag(0, 0), 0.0)

    def test_sag_finite_radii(self, set_test_backend):
        """
        Tests the sag calculation for a standard biconic surface with finite
        radii.
        """
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=10.0, radius_y=20.0)
        assert_allclose(geom.sag(x=1, y=1), 0.07514126037252641)

    def test_sag_rx_infinite(self, set_test_backend):
        """
        Tests the sag for a cylindrical surface with infinite radius in x.
        The sag should only depend on the y-coordinate.
        """
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=be.inf, radius_y=20.0)
        assert_allclose(geom.sag(x=10, y=1), 0.02501563183003138)
        assert_allclose(geom.sag(x=-5, y=1), 0.02501563183003138)

    def test_sag_ry_infinite(self, set_test_backend):
        """
        Tests the sag for a cylindrical surface with infinite radius in y.
        The sag should only depend on the x-coordinate.
        """
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=10.0, radius_y=be.inf)
        assert_allclose(geom.sag(x=1, y=10), 0.05012562854249503)
        assert_allclose(geom.sag(x=1, y=-5), 0.05012562854249503)

    def test_sag_both_infinite_plane(self, set_test_backend):
        """
        Tests that if both radii are infinite, the surface is a plane with
        zero sag everywhere.
        """
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=be.inf, radius_y=be.inf)
        assert_allclose(geom.sag(x=10, y=20), 0.0)
        geom_conic = BiconicGeometry(
            cs, radius_x=be.inf, radius_y=be.inf, conic_x=0.5, conic_y=-1.0
        )
        assert_allclose(geom_conic.sag(x=10, y=20), 0.0)

    def test_sag_with_conics(self, set_test_backend):
        """
        Tests the sag calculation for a biconic surface with non-zero conic
        constants.
        """
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=20.0, conic_x=-1.0, conic_y=0.5
        )
        expected_sag = 0.05 + 0.02502345130203264
        assert_allclose(geom.sag(x=1, y=1), expected_sag)

    def test_sag_array_input(self, set_test_backend):
        """
        Tests that the sag calculation works correctly with array inputs for
        x and y coordinates.
        """
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=10.0, radius_y=20.0)
        x = be.array([0, 1, 2])
        y = be.array([0, 1, 1])
        expected_sags = be.array([0.0, 0.07514126037252641, 0.22705672190612818])
        assert_allclose(geom.sag(x, y), expected_sags)

    def test_surface_normal_vertex(self, set_test_backend):
        """
        Tests that the surface normal at the vertex is [0, 0, -1].
        """
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=20.0, conic_x=0.5, conic_y=-0.5
        )
        nx, ny, nz = geom._surface_normal(x=0, y=0)
        assert_allclose(nx, 0.0)
        assert_allclose(ny, 0.0)
        assert_allclose(nz, -1.0)

    def test_surface_normal_spherical_case(self, set_test_backend):
        """
        Tests that when both radii and conics are equal, the surface normal
        matches that of a standard spherical surface.
        """
        cs = CoordinateSystem()
        R = 10.0
        geom = BiconicGeometry(cs, radius_x=R, radius_y=R, conic_x=0.0, conic_y=0.0)
        nx, ny, nz = geom._surface_normal(x=1, y=1)
        assert_allclose(nx, 0.09950371902099892)
        assert_allclose(ny, 0.09950371902099892)
        assert_allclose(nz, -0.9900493732390136)

    def test_surface_normal_cylindrical_rx_inf(self, set_test_backend):
        """
        Tests the surface normal for a cylindrical surface (Rx=inf). The normal
        in the x-direction should be zero.
        """
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=be.inf, radius_y=10.0, conic_y=0.0)
        nx, ny, nz = geom._surface_normal(x=5, y=1)
        assert_allclose(nx, 0.0)
        assert_allclose(ny, 0.1)
        assert_allclose(nz, -0.99498743710662)

    def test_surface_normal_array_input(self, set_test_backend):
        """
        Tests that the surface normal calculation works correctly with array
        inputs.
        """
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=10.0, conic_x=0.0, conic_y=0.0
        )
        x = be.array([0, 1])
        y = be.array([0, 1])
        expected_nx = be.array([0.0, 0.09950371902099892])
        expected_ny = be.array([0.0, 0.09950371902099892])
        expected_nz = be.array([-1.0, -0.9900493732390136])
        nx_calc, ny_calc, nz_calc = geom._surface_normal(x, y)
        assert_allclose(nx_calc, expected_nx)
        assert_allclose(ny_calc, expected_ny)
        assert_allclose(nz_calc, expected_nz)

    def test_distance_simple(self, set_test_backend):
        """
        Tests the ray intersection distance for a simple on-axis ray.
        """
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=10.0, radius_y=20.0)
        rays = RealRays(
            x=0.0, y=0.0, z=-5.0, L=0.0, M=0.0, N=1.0, wavelength=0.55, intensity=1.0
        )
        assert_allclose(geom.distance(rays), 5.0, atol=1e-9)

    def test_distance_planar_biconic(self, set_test_backend):
        """
        Tests that a biconic surface with infinite radii behaves like a plane
        for ray intersection.
        """
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=be.inf, radius_y=be.inf)
        rays = RealRays(
            x=1.0, y=1.0, z=-5.0, L=0.0, M=0.0, N=1.0, wavelength=0.55, intensity=1.0
        )
        assert_allclose(geom.distance(rays), 5.0, atol=1e-9)

    def test_to_dict_from_dict(self, set_test_backend):
        """
        Tests the serialization to and deserialization from a dictionary.
        """
        cs = CoordinateSystem(x=1, y=2, z=3, rx=0.1, ry=-0.1, rz=0.05)
        original_geom = BiconicGeometry(
            cs,
            radius_x=100.0,
            radius_y=-150.0,
            conic_x=-0.5,
            conic_y=0.2,
            tol=1e-9,
            max_iter=50,
        )
        geom_dict = original_geom.to_dict()

        assert geom_dict["type"] == "BiconicGeometry"
        assert geom_dict["radius_x"] == 100.0
        assert geom_dict["radius_y"] == -150.0
        assert geom_dict["conic_x"] == -0.5
        assert geom_dict["conic_y"] == 0.2

        reconstructed_geom = BiconicGeometry.from_dict(geom_dict)
        assert isinstance(reconstructed_geom, BiconicGeometry)
        assert reconstructed_geom.to_dict() == geom_dict

    def test_from_dict_missing_keys(self, set_test_backend):
        """
        Tests that deserializing from a dictionary with missing required keys
        raises a ValueError.
        """
        cs = CoordinateSystem()
        minimal_valid_dict = {
            "type": "BiconicGeometry",
            "cs": cs.to_dict(),
            "radius_x": 10.0,
            "radius_y": 20.0,
        }
        invalid_dict_rx = minimal_valid_dict.copy()
        del invalid_dict_rx["radius_x"]
        with pytest.raises(ValueError):
            BiconicGeometry.from_dict(invalid_dict_rx)

    def test_from_dict_default_conics_tol_max_iter(self, set_test_backend):
        """
        Tests that optional parameters (conics, tol, max_iter) are assigned
        their default values when not present in the dictionary.
        """
        cs = CoordinateSystem()
        geom_data = {
            "type": "BiconicGeometry",
            "cs": cs.to_dict(),
            "radius_x": 10.0,
            "radius_y": 20.0,
        }
        geom = BiconicGeometry.from_dict(geom_data)
        assert geom.kx == 0.0
        assert geom.ky == 0.0
        assert geom.tol == 1e-10
        assert geom.max_iter == 100