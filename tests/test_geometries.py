import pytest
import numpy as np
from optiland.rays import RealRays
from optiland.coordinate_system import CoordinateSystem
from optiland import geometries


class TestPlane:
    def test_plane_sag(self):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        # Test sag at (0, 0)
        assert plane.sag() == 0.0

        # Test sag at (1, 1)
        assert plane.sag(1, 1) == 0.0

        # Test sag at (-2, 3)
        assert plane.sag(-2, 3) == 0.0

    def test_plane_distance(self):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = plane.distance(rays)
        assert distance == pytest.approx(3.0, abs=1e-10)

        # Test distance for multiple rays
        rays = RealRays([1.0, 2.0], [2.0, 3.0], [-3.0, -4.0], [0.0, 0.0],
                        [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
        distance = plane.distance(rays)
        assert distance == pytest.approx([3.0, 4.0], abs=1e-10)

        # Test distance for ray not parallel to z axis
        L = 0.356
        M = -0.129
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -16.524, L, M, N, 1.0, 0.0)
        distance = plane.distance(rays)
        assert distance == pytest.approx(17.853374740457518, abs=1e-10)

    def test_plane_surface_normal(self):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = plane.surface_normal(rays)
        assert nx == 0.0
        assert ny == 0.0
        assert nz == 1.0
