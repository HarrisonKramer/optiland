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


class TestStandardGeometry:
    def test_sag_sphere(self):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.0)

        # Test sag at (0, 0)
        assert geometry.sag() == pytest.approx(0.0, abs=1e-10)

        # Test sag at (1, 1)
        assert geometry.sag(1, 1) == pytest.approx(0.10050506338833465,
                                                   abs=1e-10)

        # Test sag at (-2, 3)
        assert geometry.sag(-2, 3) == pytest.approx(0.6726209469111849,
                                                    abs=1e-10)

        # Test array input
        x = np.array([0, 3, 8])
        y = np.array([0, -7, 2.1])
        sag = np.array([0.0, 3.5192593015921396, 4.3795018014414415])
        assert np.allclose(geometry.sag(x, y), sag)

    def test_sag_parabola(self):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=25.0, conic=-1.0)

        # Test sag at (0, 0)
        assert geometry.sag() == pytest.approx(0.0, abs=1e-10)

        # Test sag at (2.1, -1.134)
        assert geometry.sag(2.1, -1.134) == pytest.approx(0.11391912,
                                                          abs=1e-10)

        # Test sag at (5, 5)
        assert geometry.sag(5, 5) == pytest.approx(1.0, abs=1e-10)

        # Test array input
        x = np.array([0, 2, 4])
        y = np.array([0, -3, 2.1])
        sag = np.array([0.0, 0.26, 0.4082])
        assert np.allclose(geometry.sag(x, y), sag)

    def test_sag_conic(self):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=27.0, conic=0.55)

        # Test sag at (0, 0)
        assert geometry.sag() == pytest.approx(0.0, abs=1e-10)

        # Test sag at (3.1, -3.134)
        assert geometry.sag(3.1, -3.134) == pytest.approx(0.3636467856728104,
                                                          abs=1e-10)

        # Test sag at (2, 5)
        assert geometry.sag(2, 5) == pytest.approx(0.5455809402149067,
                                                   abs=1e-10)

        # Test array input
        x = np.array([0, 5, 6])
        y = np.array([0, -3, 3.1])
        sag = np.array([0.0, 0.6414396188168761, 0.8661643140626132])
        assert np.allclose(geometry.sag(x, y), sag)

    def test_distance(self):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=-12.0, conic=0.5)

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert distance == pytest.approx(2.7888809636986154, abs=1e-10)

        # Test distance for multiple rays
        rays = RealRays([1.0, 2.0], [2.0, 3.0], [-3.0, -4.0], [0.0, 0.0],
                        [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
        distance = geometry.distance(rays)
        nom_distance = [2.7888809636986154, 3.4386378681404657]
        assert distance == pytest.approx(nom_distance, abs=1e-10)

        # Test distance for ray not parallel to z axis
        L = 0.359
        M = -0.229
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert distance == pytest.approx(10.201933401020467, abs=1e-10)

    def test_surface_normal(self):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.5)

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert nx == pytest.approx(-0.10127393670836665, abs=1e-10)
        assert ny == pytest.approx(-0.2025478734167333, abs=1e-10)
        assert nz == pytest.approx(0.9740215340114144, abs=1e-10)
