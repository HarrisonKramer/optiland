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

        # Test array input
        x = np.array([0, 3, 8e3])
        y = np.array([0, -7.0, 2.1654])
        sag = np.array([0.0, 0.0, 0.0])
        assert np.allclose(plane.sag(x, y), sag)

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

        # Test ray doesn't intersect the plane
        rays = RealRays(1.0, 2.0, -1.5, 0.0, 0.0, -1.0, 1.0, 0.0)
        distance = plane.distance(rays)
        assert np.isnan(distance)

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
        assert nx == pytest.approx(0.10127393670836665, abs=1e-10)
        assert ny == pytest.approx(0.2025478734167333, abs=1e-10)
        assert nz == pytest.approx(-0.9740215340114144, abs=1e-10)


class TestEvenAsphere:
    def test_sag(self):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(cs, radius=27.0, conic=0.0,
                                          coefficients=[1e-3, -1e-5])

        # Test sag at (0, 0)
        assert geometry.sag() == pytest.approx(0.0, abs=1e-10)

        # Test sag at (1, 1)
        assert geometry.sag(1, 1) == pytest.approx(0.039022474574473776,
                                                   abs=1e-10)

        # Test sag at (-2, 3)
        assert geometry.sag(-2, 3) == pytest.approx(0.25313367948069593,
                                                    abs=1e-10)

        # Test array input
        x = np.array([0, 3, 8])
        y = np.array([0, -7, 2.1])
        sag = np.array([0.0, 1.1206923060227627, 1.3196652673420655])
        assert np.allclose(geometry.sag(x, y), sag)

    def test_distance(self):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(cs, radius=-41.1, conic=0.0,
                                          coefficients=[1e-3, -1e-5, 1e-7])

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert distance == pytest.approx(2.9438901710409624, abs=1e-10)

        # Test distance for multiple rays
        rays = RealRays([1.0, 2.0], [2.0, 3.0], [-3.0, -4.0], [0.0, 0.0],
                        [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
        distance = geometry.distance(rays)
        nom_distance = [2.9438901710409624, 3.8530733934173256]
        assert distance == pytest.approx(nom_distance, abs=1e-10)

        # Test distance for ray not parallel to z axis
        L = 0.222
        M = -0.229
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert distance == pytest.approx(10.625463223037386, abs=1e-10)

    def test_surface_normal(self):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(cs, radius=10.0, conic=0.5,
                                          coefficients=[1e-2])

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert nx == pytest.approx(0.11946945186789681, abs=1e-10)
        assert ny == pytest.approx(0.23893890373579363, abs=1e-10)
        assert nz == pytest.approx(-0.9636572265862595, abs=1e-10)


class TestPolynomialGeometry:
    def test_sag(self):
        cs = CoordinateSystem()
        coefficients = np.zeros((3, 3))
        coefficients[0] = [0.0, 1e-2, -2e-3]
        coefficients[1] = [0.1, 1e-2, -1e-3]
        coefficients[2] = [0.2, 1e-2, 0.0]
        geometry = geometries.PolynomialGeometry(cs, radius=22.0, conic=0.0,
                                                 coefficients=coefficients)

        # Test sag at (0, 0)
        assert geometry.sag() == pytest.approx(0.0, abs=1e-10)

        # Test sag at (1, 1)
        assert geometry.sag(1, 1) == pytest.approx(0.3725015998998511,
                                                   abs=1e-10)

        # Test sag at (-2, -7)
        assert geometry.sag(-2, -7) == pytest.approx(1.6294605079733058,
                                                     abs=1e-10)

        # Test array input
        x = np.array([0, 3, 8])
        y = np.array([0, -7, 2.1])
        sag = np.array([0.0, 2.305232559449707, 16.702875375272402])
        assert np.allclose(geometry.sag(x, y), sag)

    def test_distance(self):
        cs = CoordinateSystem()
        coefficients = np.zeros((3, 3))
        coefficients[0] = [0.0, 1e-2, 2e-3]
        coefficients[1] = [0.1, -1e-2, 1e-3]
        coefficients[2] = [0.2, 1e-2, 2e-4]
        geometry = geometries.PolynomialGeometry(cs, radius=-26.0, conic=0.1,
                                                 coefficients=coefficients)

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert distance == pytest.approx(3.236449774952821, abs=1e-10)

        # Test distance for multiple rays
        rays = RealRays([1.0, 2.0], [2.0, 3.0], [-3.0, -4.0], [0.0, 0.0],
                        [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
        distance = geometry.distance(rays)
        nom_distance = [3.236449774952821, 4.881863713037335]
        assert distance == pytest.approx(nom_distance, abs=1e-10)

        # Test distance for ray not parallel to z axis
        L = 0.164
        M = -0.210
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert distance == pytest.approx(12.610897321951025, abs=1e-10)

    def test_surface_normal(self):
        cs = CoordinateSystem()
        coefficients = np.zeros((3, 3))
        coefficients[0] = [0.0, 1e-2, 2e-3]
        coefficients[1] = [0.1, -1e-2, 1e-3]
        coefficients[2] = [0.2, 1e-2, 2e-4]
        geometry = geometries.PolynomialGeometry(cs, radius=-26.0, conic=0.1,
                                                 coefficients=coefficients)

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert nx == pytest.approx(0.4373017765693584, abs=1e-10)
        assert ny == pytest.approx(-0.04888445345459283, abs=1e-10)
        assert nz == pytest.approx(-0.8979852261700794, abs=1e-10)


class TestChebyshevGeometry:
    def test_sag(self):
        cs = CoordinateSystem()
        coefficients = np.zeros((3, 3))
        coefficients[0] = [0.0, 1e-2, -2e-3]
        coefficients[1] = [0.1, 1e-2, -1e-3]
        coefficients[2] = [0.2, 1e-2, 0.0]
        geometry = \
            geometries.ChebyshevPolynomialGeometry(cs, radius=22.0, conic=0.0,
                                                   coefficients=coefficients,
                                                   norm_x=10, norm_y=10)

        # Test sag at (0, 0)
        assert geometry.sag() == pytest.approx(-0.198, abs=1e-10)

        # Test sag at (1, 1)
        assert geometry.sag(1, 1) == pytest.approx(-0.13832040010014895,
                                                   abs=1e-10)

        # Test sag at (-2, -7)
        assert geometry.sag(-2, -7) == pytest.approx(1.036336507973306,
                                                     abs=1e-10)

        # Test array input
        x = np.array([0, 3, 8])
        y = np.array([0, -7, 2.1])
        sag = np.array([-0.198, 1.22291856, 1.75689642])
        assert np.allclose(geometry.sag(x, y), sag)

    def test_distance(self):
        cs = CoordinateSystem()
        coefficients = np.zeros((3, 3))
        coefficients[0] = [0.0, 1e-2, -2e-3]
        coefficients[1] = [0.1, 1e-2, -1e-3]
        coefficients[2] = [0.2, 1e-2, 0.0]
        geometry = \
            geometries.ChebyshevPolynomialGeometry(cs, radius=-26.0, conic=0.1,
                                                   coefficients=coefficients,
                                                   norm_x=10, norm_y=10)

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert distance == pytest.approx(2.71982177, abs=1e-8)

        # Test distance for multiple rays
        rays = RealRays([1.0, 2.0], [2.0, 3.0], [-3.0, -4.0], [0.0, 0.0],
                        [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
        distance = geometry.distance(rays)
        nom_distance = [2.719821774952821, 3.5873077130373345]
        assert distance == pytest.approx(nom_distance, abs=1e-8)

        # Test distance for ray not parallel to z axis
        L = 0.164
        M = -0.210
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert distance == pytest.approx(10.29015593, abs=1e-8)

    def test_surface_normal(self):
        cs = CoordinateSystem()
        coefficients = np.zeros((3, 3))
        coefficients[0] = [0.0, 1e-2, -2e-3]
        coefficients[1] = [0.1, 1e-2, -1e-3]
        coefficients[2] = [0.2, 1e-2, 0.0]
        geometry = \
            geometries.ChebyshevPolynomialGeometry(cs, radius=-26.0, conic=0.1,
                                                   coefficients=coefficients,
                                                   norm_x=10, norm_y=10)

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert nx == pytest.approx(0.14317439, abs=1e-8)
        assert ny == pytest.approx(-0.07668599, abs=1e-8)
        assert nz == pytest.approx(-0.98672202, abs=1e-8)

    def test_invalid_input(self):
        cs = CoordinateSystem()
        coefficients = np.zeros((3, 3))
        coefficients[0] = [0.0, 1e-2, -2e-3]
        coefficients[1] = [0.1, 1e-2, -1e-3]
        coefficients[2] = [0.2, 1e-2, 0.0]
        geometry = \
            geometries.ChebyshevPolynomialGeometry(cs, radius=-26.0, conic=0.1,
                                                   coefficients=coefficients,
                                                   norm_x=10, norm_y=10)

        with pytest.raises(ValueError):
            geometry.sag(100, 100)
