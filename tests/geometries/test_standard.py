# tests/geometries/test_standard.py
"""
Tests for the StandardGeometry class in optiland.geometries.
"""
import pytest
import numpy as np

import optiland.backend as be
from optiland import geometries
from optiland.coordinate_system import CoordinateSystem
from optiland.rays import RealRays
from ..utils import assert_allclose


class TestStandardGeometry:
    """
    Tests for the StandardGeometry class, which represents a standard conic
    surface.
    """

    def test_str(self, set_test_backend):
        """
        Tests the string representation of the StandardGeometry.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.5)
        assert str(geometry) == "Standard"

    def test_sag_sphere(self, set_test_backend):
        """
        Tests the sag calculation for a spherical surface (conic = 0).
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.0)

        # Test sag at the origin
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at various points
        assert_allclose(geometry.sag(1, 1), 0.10050506338833465)
        assert_allclose(geometry.sag(-2, 3), 0.6726209469111849)

        # Test with array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([0.0, 3.5192593015921396, 4.3795018014414415])
        assert_allclose(geometry.sag(x, y), sag)

    def test_sag_parabola(self, set_test_backend):
        """
        Tests the sag calculation for a parabolic surface (conic = -1).
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=25.0, conic=-1.0)

        # Test sag at the origin
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at various points
        assert_allclose(geometry.sag(2.1, -1.134), 0.11391912)
        assert_allclose(geometry.sag(5, 5), 1.0)

        # Test with array input
        x = be.array([0, 2, 4])
        y = be.array([0, -3, 2.1])
        sag = be.array([0.0, 0.26, 0.4082])
        assert_allclose(geometry.sag(x, y), sag)

    def test_sag_conic(self, set_test_backend):
        """
        Tests the sag calculation for a general conic surface.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=27.0, conic=0.55)

        # Test sag at the origin
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at various points
        assert_allclose(geometry.sag(3.1, -3.134), 0.3636467856728104)
        assert_allclose(geometry.sag(2, 5), 0.5455809402149067)

        # Test with array input
        x = be.array([0, 5, 6])
        y = be.array([0, -3, 3.1])
        sag = be.array([0.0, 0.6414396188168761, 0.8661643140626132])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        """
        Tests the calculation of the distance from a ray's origin to its
        intersection point with the surface.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=-12.0, conic=0.5)

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 2.7888809636986154)

        # Test distance for multiple rays
        rays = RealRays(
            [1.0, 2.0],
            [2.0, 3.0],
            [-3.0, -4.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        )
        distance = geometry.distance(rays)
        nom_distance = [2.7888809636986154, 3.4386378681404657]
        assert_allclose(distance, nom_distance)

        # Test distance for a ray not parallel to the z-axis
        L = 0.359
        M = -0.229
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 10.201933401020467)

    def test_surface_normal(self, set_test_backend):
        """
        Tests the calculation of the surface normal vector at a given point.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.5)
        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert_allclose(nx, 0.10127393670836665)
        assert_allclose(ny, 0.2025478734167333)
        assert_allclose(nz, -0.9740215340114144)

    def test_to_dict(self, set_test_backend):
        """
        Tests the serialization of a StandardGeometry instance to a dictionary.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.5)
        expected_dict = {
            "type": "StandardGeometry",
            "cs": cs.to_dict(),
            "radius": 10.0,
            "conic": 0.5,
        }
        assert geometry.to_dict() == expected_dict

    def test_from_dict(self, set_test_backend):
        """
        Tests the deserialization of a StandardGeometry instance from a
        dictionary.
        """
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.5)
        geometry_dict = geometry.to_dict()
        new_geometry = geometries.StandardGeometry.from_dict(geometry_dict)
        assert new_geometry.to_dict() == geometry_dict

    def test_From_dict_invalid_dict(self, set_test_backend):
        """
        Tests that attempting to deserialize from an invalid dictionary
        raises a ValueError.
        """
        with pytest.raises(ValueError):
            geometries.StandardGeometry.from_dict({"invalid_key": "invalid_value"})