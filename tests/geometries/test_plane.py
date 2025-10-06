# tests/geometries/test_plane.py
"""
Tests for the Plane geometry class in optiland.geometries.
"""
import optiland.backend as be
import numpy as np

from optiland import geometries
from optiland.coordinate_system import CoordinateSystem
from optiland.rays import RealRays
from ..utils import assert_allclose


class TestPlane:
    """
    Tests for the Plane geometry, which represents a perfectly flat surface.
    """

    def test_str(self, set_test_backend):
        """
        Tests the string representation of the Plane geometry.
        """
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)
        assert str(plane) == "Planar"

    def test_plane_sag(self, set_test_backend):
        """
        Tests that the sag of a plane is always zero, regardless of the
        (x, y) coordinates.
        """
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        # Test sag at the origin
        assert plane.sag() == 0.0

        # Test sag at an arbitrary point
        assert plane.sag(1, 1) == 0.0
        assert plane.sag(-2, 3) == 0.0

        # Test with array input
        x = be.array([0, 3, 8e3])
        y = be.array([0, -7.0, 2.1654])
        sag = be.array([0.0, 0.0, 0.0])
        assert be.allclose(plane.sag(x, y), sag)

    def test_plane_distance(self, set_test_backend):
        """
        Tests the calculation of the distance from a ray's origin to its
        intersection point with the plane.
        """
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        # Test distance for a single ray parallel to the z-axis
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = plane.distance(rays)
        assert_allclose(distance, 3.0)

        # Test distance for multiple parallel rays
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
        distance = plane.distance(rays)
        assert_allclose(distance, [3.0, 4.0])

        # Test a ray traveling away from the plane (negative distance)
        rays = RealRays(1.0, 2.0, -1.5, 0.0, 0.0, -1.0, 1.0, 0.0)
        distance = plane.distance(rays)
        assert_allclose(distance, -1.5)

        # Test distance for a ray not parallel to the z-axis
        L = 0.356
        M = -0.129
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -16.524, L, M, N, 1.0, 0.0)
        distance = plane.distance(rays)
        assert_allclose(distance, 17.853374740457518)

    def test_plane_surface_normal(self, set_test_backend):
        """
        Tests that the surface normal of a plane is always [0, 0, 1] in its
        local coordinate system.
        """
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)
        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = plane.surface_normal(rays)
        assert nx == 0.0
        assert ny == 0.0
        assert nz == 1.0

    def test_to_dict(self, set_test_backend):
        """
        Tests the serialization of a Plane instance to a dictionary.
        """
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)
        expected_dict = {"type": "Plane", "cs": cs.to_dict(), "radius": be.inf}
        assert plane.to_dict() == expected_dict

    def test_from_dict(self, set_test_backend):
        """
        Tests the deserialization of a Plane instance from a dictionary.
        """
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)
        plane_dict = plane.to_dict()
        new_plane = geometries.Plane.from_dict(plane_dict)
        assert new_plane.to_dict() == plane_dict