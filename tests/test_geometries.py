import optiland.backend as be
import pytest
import numpy as np
from optiland.optic import Optic
from optiland.materials import Material, IdealMaterial
from optiland import geometries
from optiland.geometries import BiconicGeometry
from optiland.coordinate_system import CoordinateSystem
from optiland.rays import RealRays
from .utils import assert_allclose


def test_unknown_geometry(set_test_backend):
    with pytest.raises(ValueError):
        geometries.BaseGeometry.from_dict({"type": "UnknownGeometry"})


class TestPlane:
    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)
        assert str(plane) == "Planar"

    def test_plane_sag(self, set_test_backend):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        # Test sag at (0, 0)
        assert plane.sag() == 0.0

        # Test sag at (1, 1)
        assert plane.sag(1, 1) == 0.0

        # Test sag at (-2, 3)
        assert plane.sag(-2, 3) == 0.0

        # Test array input
        x = be.array([0, 3, 8e3])
        y = be.array([0, -7.0, 2.1654])
        sag = be.array([0.0, 0.0, 0.0])
        assert be.allclose(plane.sag(x, y), sag)

    def test_plane_distance(self, set_test_backend):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = plane.distance(rays)
        assert_allclose(distance, 3.0)

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
        distance = plane.distance(rays)
        assert_allclose(distance, [3.0, 4.0])

        # Test ray doesn't intersect the plane
        rays = RealRays(1.0, 2.0, -1.5, 0.0, 0.0, -1.0, 1.0, 0.0)
        distance = plane.distance(rays)
        assert_allclose(distance, -1.5)

        # Test distance for ray not parallel to z axis
        L = 0.356
        M = -0.129
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -16.524, L, M, N, 1.0, 0.0)
        distance = plane.distance(rays)
        assert_allclose(distance, 17.853374740457518)

    def test_plane_surface_normal(self, set_test_backend):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = plane.surface_normal(rays)
        assert nx == 0.0
        assert ny == 0.0
        assert nz == 1.0

    def test_to_dict(self, set_test_backend):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        expected_dict = {"type": "Plane", "cs": cs.to_dict(), "radius": be.inf}
        assert plane.to_dict() == expected_dict

    def test_from_dict(self, set_test_backend):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        plane_dict = plane.to_dict()
        new_plane = geometries.Plane.from_dict(plane_dict)
        assert new_plane.to_dict() == plane_dict


class TestStandardGeometry:
    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.5)
        assert str(geometry) == "Standard"

    def test_sag_sphere(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.0)

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at (1, 1)
        assert_allclose(geometry.sag(1, 1), 0.10050506338833465)

        # Test sag at (-2, 3)
        assert_allclose(geometry.sag(-2, 3), 0.6726209469111849)

        # Test array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([0.0, 3.5192593015921396, 4.3795018014414415])
        assert_allclose(geometry.sag(x, y), sag)

    def test_sag_parabola(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=25.0, conic=-1.0)

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at (2.1, -1.134)
        assert_allclose(geometry.sag(2.1, -1.134), 0.11391912)

        # Test sag at (5, 5)
        assert_allclose(geometry.sag(5, 5), 1.0)

        # Test array input
        x = be.array([0, 2, 4])
        y = be.array([0, -3, 2.1])
        sag = be.array([0.0, 0.26, 0.4082])
        assert_allclose(geometry.sag(x, y), sag)

    def test_sag_conic(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=27.0, conic=0.55)

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at (3.1, -3.134)
        assert_allclose(geometry.sag(3.1, -3.134), 0.3636467856728104)

        # Test sag at (2, 5)
        assert_allclose(geometry.sag(2, 5), 0.5455809402149067)

        # Test array input
        x = be.array([0, 5, 6])
        y = be.array([0, -3, 3.1])
        sag = be.array([0.0, 0.6414396188168761, 0.8661643140626132])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
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

        # Test distance for ray not parallel to z axis
        L = 0.359
        M = -0.229
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 10.201933401020467)

    def test_surface_normal(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.5)

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert_allclose(nx, 0.10127393670836665)
        assert_allclose(ny, 0.2025478734167333)
        assert_allclose(nz, -0.9740215340114144)

    def test_to_dict(self, set_test_backend):
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
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.5)

        geometry_dict = geometry.to_dict()
        new_geometry = geometries.StandardGeometry.from_dict(geometry_dict)
        assert new_geometry.to_dict() == geometry_dict

    def test_From_dict_invalid_dict(self, set_test_backend):
        with pytest.raises(ValueError):
            geometries.StandardGeometry.from_dict({"invalid_key": "invalid_value"})


class TestEvenAsphere:
    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2],
        )
        assert str(geometry) == "Even Asphere"

    def test_sag(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=27.0,
            conic=0.0,
            coefficients=[1e-3, -1e-5],
        )

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at (1, 1)
        assert_allclose(geometry.sag(1, 1), 0.039022474574473776)

        # Test sag at (-2, 3)
        assert_allclose(geometry.sag(-2, 3), 0.25313367948069593)

        # Test array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([0.0, 1.1206923060227627, 1.3196652673420655])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=-41.1,
            conic=0.0,
            coefficients=[1e-3, -1e-5, 1e-7],
        )

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 2.9438901710409624)

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
        nom_distance = [2.9438901710409624, 3.8530733934173256]
        assert_allclose(distance, nom_distance)

        # Test distance for ray not parallel to z axis
        L = 0.222
        M = -0.229
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 10.625463223037386)

    def test_surface_normal(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2],
        )

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert_allclose(nx, 0.11946945186789681)
        assert_allclose(ny, 0.23893890373579363)
        assert_allclose(nz, -0.9636572265862595)

    def test_to_dict(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2],
        )

        expected_dict = {
            "type": "EvenAsphere",
            "cs": cs.to_dict(),
            "radius": 10.0,
            "conic": 0.5,
            "tol": 1e-10,
            "max_iter": 100,
            "coefficients": [1e-2],
        }
        assert geometry.to_dict() == expected_dict

    def test_from_dict(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2],
        )

        geometry_dict = geometry.to_dict()
        new_geometry = geometries.EvenAsphere.from_dict(geometry_dict)
        assert new_geometry.to_dict() == geometry_dict


class TestPolynomialGeometry:
    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.PolynomialGeometry(
            cs,
            radius=22.0,
            conic=0.0,
            coefficients=coefficients,
        )
        assert str(geometry) == "Polynomial XY"

    def test_sag(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.PolynomialGeometry(
            cs,
            radius=22.0,
            conic=0.0,
            coefficients=coefficients,
        )

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at (1, 1)
        assert_allclose(geometry.sag(1, 1), 0.3725015998998511)

        # Test sag at (-2, -7)
        assert_allclose(geometry.sag(-2, -7), 1.6294605079733058)

        # Test array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([0.0, 2.305232559449707, 16.702875375272402])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, 2e-3], [0.1, -1e-2, 1e-3], [0.2, 1e-2, 2e-4]]
        )
        geometry = geometries.PolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
        )

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 3.236449774952821)

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
        nom_distance = [3.236449774952821, 4.881863713037335]
        assert_allclose(distance, nom_distance)

        # Test distance for ray not parallel to z axis
        L = 0.164
        M = -0.210
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 12.610897321951025)

    def test_surface_normal(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, 2e-3], [0.1, -1e-2, 1e-3], [0.2, 1e-2, 2e-4]]
        )
        geometry = geometries.PolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
        )

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert_allclose(nx, 0.4373017765693584)
        assert_allclose(ny, -0.04888445345459283)
        assert_allclose(nz, -0.8979852261700794)

    def test_to_dict(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, 2e-3], [0.1, -1e-2, 1e-3], [0.2, 1e-2, 2e-4]]
        )
        geometry = geometries.PolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
        )

        expected_dict = {
            "type": "PolynomialGeometry",
            "cs": cs.to_dict(),
            "radius": -26.0,
            "conic": 0.1,
            "tol": 1e-10,
            "max_iter": 100,
            "coefficients": coefficients.tolist(),
        }
        assert geometry.to_dict() == expected_dict

    def test_from_dict(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, 2e-3], [0.1, -1e-2, 1e-3], [0.2, 1e-2, 2e-4]]
        )
        geometry = geometries.PolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
        )

        geometry_dict = geometry.to_dict()
        new_geometry = geometries.PolynomialGeometry.from_dict(geometry_dict)
        assert new_geometry.to_dict() == geometry_dict


class TestChebyshevGeometry:
    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=22.0,
            conic=0.0,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )
        assert str(geometry) == "Chebyshev Polynomial"

    def test_sag(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=22.0,
            conic=0.0,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), -0.198)

        # Test sag at (1, 1)
        assert_allclose(geometry.sag(1, 1), -0.13832040010014895)

        # Test sag at (-2, -7)
        assert_allclose(geometry.sag(-2, -7), 1.036336507973306)

        # Test array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([-0.198, 1.22291856, 1.75689642])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 2.71982177)

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
        nom_distance = [2.719821774952821, 3.5873077130373345]
        assert_allclose(distance, nom_distance)

        # Test distance for ray not parallel to z axis
        L = 0.164
        M = -0.210
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 10.29015593)

    def test_surface_normal(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert_allclose(nx, 0.14317439)
        assert_allclose(ny, -0.07668599)
        assert_allclose(nz, -0.98672202)

    def test_invalid_input(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )

        with pytest.raises(ValueError):
            geometry.sag(100, 100)

    def test_to_dict(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )

        expected_dict = {
            "type": "ChebyshevPolynomialGeometry",
            "cs": cs.to_dict(),
            "radius": -26.0,
            "conic": 0.1,
            "tol": 1e-10,
            "max_iter": 100,
            "coefficients": coefficients.tolist(),
            "norm_x": 10,
            "norm_y": 10,
        }
        assert geometry.to_dict() == expected_dict

    def test_from_dict(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )

        geometry_dict = geometry.to_dict()
        new_geometry = geometries.ChebyshevPolynomialGeometry.from_dict(geometry_dict)
        assert new_geometry.to_dict() == geometry_dict


class TestOddAsphere:
    def test_sag(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.OddAsphere(
            cs,
            radius=27.0,
            conic=0.0,
            coefficients=[1e-3, -1e-5],
        )

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at (1, 1)
        assert_allclose(geometry.sag(1, 1), 0.03845668813684687)

        # Test sag at (-2, 3)
        assert_allclose(geometry.sag(-2, 3), 0.24529923075615997)

        # Test array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([0.0, 1.10336808, 1.30564148])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.OddAsphere(
            cs,
            radius=-41.1,
            conic=0.0,
            coefficients=[1e-3, -1e-5, 1e-7],
        )

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 2.94131486)

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
        assert_allclose(distance, [2.94131486, 3.84502393])

    def test_surface_normal(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.OddAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2, -1e-5],
        )

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert_allclose(nx, 0.10537434)
        assert_allclose(ny, 0.21074867)
        assert_allclose(nz, -0.9718442)

    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.OddAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2, -1e-5],
        )
        assert str(geometry) == "Odd Asphere"


class TestZernikeGeometry:
    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array([[0.0, 1e-2, -2e-3], [0, 0, 0], [0, 0, 0]])
        geometry = geometries.ZernikePolynomialGeometry(
            cs,
            radius=22.0,
            conic=0.0,
            coefficients=coefficients,
            norm_radius=10,
        )
        assert str(geometry) == "Zernike Polynomial"


# --- Fixtures for Toroidal Tests ---
@pytest.fixture
def basic_toroid_geometry(set_test_backend):
    """Provides a basic ToroidalGeometry instance for testing"""
    cs = CoordinateSystem(x=0, y=0, z=0)
    radius_rotation = 100.0  # R (X-Z radius)
    radius_yz = 50.0  # R_y (Y-Z radius)
    conic = -0.5  # k_yz (YZ conic)
    coeffs_poly_y = [1e-5]
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_rotation=radius_rotation,
        radius_yz=radius_yz,
        conic=conic,
        coeffs_poly_y=coeffs_poly_y,
    )


@pytest.fixture
def cylinder_x_geometry(set_test_backend):
    """Provides a cylindrical geometry (flat in X). R_rot = inf"""
    cs = CoordinateSystem(x=0, y=0, z=0)
    radius_rotation = be.inf  # Flat in X-Z plane
    radius_yz = -50.0
    conic = 0.0
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_rotation=radius_rotation,
        radius_yz=radius_yz,
        conic=conic,
    )


@pytest.fixture
def cylinder_y_geometry(set_test_backend):
    """Provides a cylindrical geometry (flat in Y). R_yz = inf"""
    cs = CoordinateSystem(x=0, y=0, z=0)
    radius_rotation = 100.0
    radius_yz = be.inf  # Flat in Y-Z plane
    conic = 0.0
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_rotation=radius_rotation,
        radius_yz=radius_yz,
        conic=conic,
    )


@pytest.fixture
def toroid_no_x_curvature_coeffs(set_test_backend):
    """Toroidal surface with no curvature in x, conic_yz = -1.1, and coeffs."""
    cs = CoordinateSystem(x=0, y=0, z=0)
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_rotation=be.inf,
        radius_yz=50.0,
        conic=-1.1,
        coeffs_poly_y=[1e-5, -2e-6],
    )


@pytest.fixture
def toroid_no_y_curvature_coeffs(set_test_backend):
    """Toroidal surface with no curvature in y, conic_yz = -0.9, and coeffs.
    This a trick test to check if the conic is not used when R_yz = inf.
    """
    cs = CoordinateSystem(x=0, y=0, z=0)
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_rotation=100.0,
        radius_yz=be.inf,  # No curvature in Y (base YZ curve is flat before polynomials)
        conic=-0.9,  # This k_yz will not be used if R_yz is inf for conic part
        coeffs_poly_y=[1e-5, -2e-6],
    )


class TestToroidalGeometry:
    def test_toroidal_str(self, set_test_backend):
        """Test string representation."""
        cs = CoordinateSystem()
        geometry = geometries.ToroidalGeometry(
            cs,
            radius_rotation=100.0,
            radius_yz=50.0,
            conic=-0.5,
            coeffs_poly_y=[1e-5],
        )
        assert str(geometry) == "Toroidal"

    def test_toroidal_sag_vertex(self, basic_toroid_geometry, set_test_backend):
        """Test sag at the vertex (0, 0). Should be 0 by definition."""
        x = be.array([0.0])
        y = be.array([0.0])
        be.allclose(
            basic_toroid_geometry.sag(x, y), be.array([0.0]), rtol=1e-5, atol=1e-6
        )

    def test_toroidal_normal_vertex(self, basic_toroid_geometry, set_test_backend):
        """ " Test the normal vector at the vertex (0, 0). Should be (0, 0, -1)"""
        x = be.array([0.0])
        y = be.array([0.0])
        nx, ny, nz = basic_toroid_geometry._surface_normal(x, y)
        be.allclose(nx, be.array([0.0]), rtol=1e-5, atol=1e-6)
        be.allclose(ny, be.array([0.0]), rtol=1e-5, atol=1e-6)
        be.allclose(nz, be.array([-1.0]), rtol=1e-5, atol=1e-6)

    def test_toroidal_sag_known_points(self, basic_toroid_geometry, set_test_backend):
        """Test sag at specific points"""
        x = be.array([0.0, 10.0, 0.0])
        y = be.array([10.0, 0.0, 5.0])

        expected_z_0_10 = 1.00605051
        expected_z_10_0 = 0.50125628
        expected_z_0_5 = 0.25056330

        expected_z = be.array([expected_z_0_10, expected_z_10_0, expected_z_0_5])
        calculated_z = basic_toroid_geometry.sag(x, y)
        assert be.allclose(calculated_z, expected_z, rtol=1e-5, atol=1e-6)

    def test_toroidal_normal_known_points(
        self, basic_toroid_geometry, set_test_backend
    ):
        """Test normal at specific points."""
        x = be.array([0.0, 10.0])
        y = be.array([10.0, 0.0])

        expected_nx_0_10 = 0.0
        expected_ny_0_10 = 0.198219
        expected_nz_0_10 = -0.980158
        expected_nx_10_0 = 0.10000
        expected_ny_10_0 = 0.0
        expected_nz_10_0 = -0.994987

        expected_nx = be.array([expected_nx_0_10, expected_nx_10_0])
        expected_ny = be.array([expected_ny_0_10, expected_ny_10_0])
        expected_nz = be.array([expected_nz_0_10, expected_nz_10_0])

        nx, ny, nz = basic_toroid_geometry._surface_normal(x, y)

        rtol = 1e-5
        atol = 1e-6
        assert be.allclose(nx, expected_nx, rtol=rtol, atol=atol)
        assert be.allclose(ny, expected_ny, rtol=rtol, atol=atol)
        assert be.allclose(nz, expected_nz, rtol=rtol, atol=atol)

    def test_cylinder_x_sag(self, cylinder_x_geometry, set_test_backend):
        """Test sag for cylinder flat in X. Should only depend on y."""
        x = be.array([0.0, 10.0, 10.0])
        y = be.array([5.0, 5.0, 0.0])

        expected_z = be.array([-0.25062818, -0.25062818, 0.0])
        calculated_z = cylinder_x_geometry.sag(x, y)
        assert_allclose(calculated_z, expected_z, rtol=1e-5, atol=1e-6)

    def test_cylinder_y_sag(self, cylinder_y_geometry, set_test_backend):
        """Test sag for cylinder flat in Y. Should only depend on x."""
        x = be.array([5.0, 5.0, 0.0])
        y = be.array([0.0, 10.0, 10.0])

        expected_z = be.array([0.12507822, 0.12507822, 0.0])
        calculated_z = cylinder_y_geometry.sag(x, y)
        assert_allclose(calculated_z, expected_z, rtol=1e-5, atol=1e-6)

    def test_toroid_no_x_curvature_sag(
        self, toroid_no_x_curvature_coeffs, set_test_backend
    ):
        """Test sag for toroid with R_rot = inf and YZ coeffs."""

        geom = toroid_no_x_curvature_coeffs
        x = be.array([0.0, 10.0, -5.0])
        y = be.array([0.0, 1.0, 2.0])

        expected_z = be.array([0.0, 0.010007900001, 0.0400064001])
        calculated_z = geom.sag(x, y)
        assert_allclose(calculated_z, expected_z, rtol=1e-5, atol=1e-7)

    def test_toroid_no_y_curvature_sag(
        self, toroid_no_y_curvature_coeffs, set_test_backend
    ):
        """Test sag for toroid with R_yz = inf and YZ coeffs. Sag comes from X-rotation of YZ poly."""

        geom = toroid_no_y_curvature_coeffs
        x = be.array([0.0, 10.0, 5.0])
        y = be.array([0.0, 1.0, 2.0])

        expected_z = be.array([0.0, 0.501264335, 0.12508686])
        calculated_z = geom.sag(x, y)
        assert_allclose(calculated_z, expected_z, rtol=1e-5, atol=1e-7)

    def test_toroidal_sag_vs_zemax(self, basic_toroid_geometry, set_test_backend):
        """
        Compares sag values calculated by Optiland with
        Zemax data for the basic_toroid_geometry.
        """
        geometry = basic_toroid_geometry

        x_coords = be.array([0.0, 2.5, 0.0, -2.5, 5.0, -5.0, 2.5, -2.5])
        y_coords = be.array([0.0, 0.0, 2.5, 0.0, 2.5, -2.5, -2.5, 2.5])

        # --- Zemax Sag Data ---
        zemax_z_sag = be.array(
            [
                0.0,  # (0, 0) - Vertex
                3.125488433897521e-002,  # (2.5, 0)
                6.258204346657634e-002,  # (0, 2.5)
                3.125488433897521e-002,  # (-2.5, 0)
                1.877386899843393e-001,  # (5, 2.5)
                1.877386899843393e-001,  # (-5, -2.5)
                9.385650612446353e-002,  # (2.5, -2.5)
                9.385650612446353e-002,  # (-2.5, 2.5)
            ]
        )

        # Calculate sag using Optiland
        optiland_z_sag = geometry.sag(x_coords, y_coords)

        assert be.allclose(optiland_z_sag, zemax_z_sag, rtol=1e-5, atol=1e-6)

    def test_toroidal_ray_tracing_comparison(self, set_test_backend):
        """
        Traces rays through a single toroidal surface and compares output
        with Zemax ray tracing data for the same system.
        """
        # --- System Setup ---
        lens = Optic()
        lens.add_surface(index=0, thickness=be.inf)
        lens.add_surface(
            index=1,
            surface_type="toroidal",
            thickness=5.0,
            material=IdealMaterial(n=1.5, k=0),
            is_stop=True,
            radius=100.0,
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

        num_rays = 5  # Number of rays per fan
        wavelength = 0.550
        z_start = 0.0

        # --- Tangential (Y) Fan Test ---
        y_coords = be.linspace(-5.0, 5.0, num_rays)
        x_in_yfan = be.zeros(num_rays)
        y_in_yfan = y_coords
        z_in_yfan = be.array([z_start] * num_rays)
        L_in_yfan = be.zeros(num_rays)
        M_in_yfan = be.zeros(num_rays)
        N_in_yfan = be.ones(num_rays)
        intensity_yfan = be.ones(num_rays)
        rays_in_yfan = RealRays(
            x=x_in_yfan,
            y=y_in_yfan,
            z=z_in_yfan,
            L=L_in_yfan,
            M=M_in_yfan,
            N=N_in_yfan,
            wavelength=wavelength,
            intensity=intensity_yfan,
        )

        # Trace Y-Fan Rays
        rays_out_yfan = lens.surface_group.trace(rays_in_yfan)
        print("Y-Fan Rays:")
        print(rays_out_yfan.y)
        zemax_x_out_yfan = be.array([0.0] * num_rays)
        zemax_y_out_yfan = be.array(
            [
                -8.123193233401276e-001,
                -4.676255499616224e-001,
                0,
                4.676255499616224e-001,
                8.123193233401276e-001,
            ]
        )
        zemax_z_out_yfan = be.array([15.0] * num_rays)
        zemax_L_out_yfan = be.array([0.0] * num_rays)
        zemax_M_out_yfan = be.array(
            [
                3.251509839270260e-001,
                1.537950377308984e-001,
                0.0,
                -1.537950377308984e-001,
                -3.251509839270260e-001,
            ]
        )
        zemax_N_out_yfan = be.array(
            [
                9.456621160072382e-001,
                9.881027711576116e-001,
                1.0,
                9.881027711576116e-001,
                9.456621160072382e-001,
            ]
        )

        # Comparison Assertions for Y-Fan
        assert be.allclose(rays_out_yfan.x, zemax_x_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.y, zemax_y_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.z, zemax_z_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.L, zemax_L_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.M, zemax_M_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.N, zemax_N_out_yfan, rtol=1e-5, atol=1e-6)

        # --- Sagittal (X) Fan Test ---
        x_coords = be.linspace(-5.0, 5.0, num_rays)
        x_in_xfan = x_coords
        y_in_xfan = be.zeros(num_rays)
        z_in_xfan = be.array([z_start] * num_rays)
        L_in_xfan = be.zeros(num_rays)
        M_in_xfan = be.zeros(num_rays)
        N_in_xfan = be.ones(num_rays)
        intensity_xfan = be.ones(num_rays)
        rays_in_xfan = RealRays(
            x=x_in_xfan,
            y=y_in_xfan,
            z=z_in_xfan,
            L=L_in_xfan,
            M=M_in_xfan,
            N=N_in_xfan,
            wavelength=wavelength,
            intensity=intensity_xfan,
        )

        rays_out_xfan = lens.surface_group.trace(rays_in_xfan)

        zemax_x_out_xfan = be.array(
            [
                -4.668385225648558e000,
                -2.333547899735358e000,
                0.0,
                2.333547899735358e000,
                4.668385225648558e000,
            ]
        )
        zemax_y_out_xfan = be.array([0.0] * num_rays)
        zemax_z_out_xfan = be.array([15.0] * num_rays)
        zemax_L_out_xfan = be.array(
            [
                2.502086086422164e-002,
                1.250260502601134e-002,
                0.0,
                -1.250260502601134e-002,
                -2.502086086422164e-002,
            ]
        )
        zemax_M_out_xfan = be.array([0.0] * num_rays)
        zemax_N_out_xfan = be.array(
            [
                9.996869292541608e-001,
                9.999218393792406e-001,
                1.0,
                9.999218393792406e-001,
                9.996869292541608e-001,
            ]
        )

        assert be.allclose(rays_out_xfan.x, zemax_x_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.y, zemax_y_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.z, zemax_z_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.L, zemax_L_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.M, zemax_M_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.N, zemax_N_out_xfan, rtol=1e-5, atol=1e-6)

    def test_toroidal_to_dict(self, basic_toroid_geometry, set_test_backend):
        """Test serialization to dictionary."""
        geom_dict = basic_toroid_geometry.to_dict()
        assert geom_dict["type"] == "ToroidalGeometry"
        assert geom_dict["geometry_type"] == "Toroidal"
        assert geom_dict["radius_rotation"] == 100.0
        assert geom_dict["radius_yz"] == 50.0
        assert geom_dict["conic_yz"] == -0.5
        assert geom_dict["coeffs_poly_y"] == pytest.approx([1e-5])

        assert "radius" not in geom_dict
        assert "conic" not in geom_dict
        assert "coefficients" not in geom_dict

    def test_toroidal_from_dict(self, basic_toroid_geometry, set_test_backend):
        """Test deserialization from dictionary."""
        geom_dict = basic_toroid_geometry.to_dict()
        new_geometry = geometries.ToroidalGeometry.from_dict(geom_dict)
        assert isinstance(new_geometry, geometries.ToroidalGeometry)
        assert new_geometry.to_dict() == geom_dict

    def test_toroidal_from_dict_invalid(self, set_test_backend):
        """Test deserialization with missing keys."""
        cs = CoordinateSystem()
        invalid_dict = {
            "type": "ToroidalGeometry",
            "cs": cs.to_dict(),
            # Missing radius_rotation, radius_yz
        }
        with pytest.raises(ValueError):
            geometries.ToroidalGeometry.from_dict(invalid_dict)

    def test_inf_radius_intersect_sphere_normal_incidence(
        self, cylinder_x_geometry, set_test_backend
    ):
        """Test _intersection_sphere for inf radius: ray normal to XY plane (z=0 plane)."""

        rays = RealRays(
            x=0.0, y=0.0, z=10.0, L=0.0, M=0.0, N=-1.0, intensity=1.0, wavelength=0.55
        )
        ix, iy, iz = cylinder_x_geometry._intersection(rays)

        be.allclose(ix, be.array(0.0), rtol=1e-5, atol=1e-6)
        be.allclose(iy, be.array(0.0), rtol=1e-5, atol=1e-6)
        be.allclose(
            iz, be.array(0.0), rtol=1e-5, atol=1e-6
        )  # Intersection coordinates should be (0, 0, 0)


class TestBiconicGeometry:
    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=10.0, radius_y=20.0)
        assert str(geom) == "Biconic"

    def test_sag_vertex(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=20.0, conic_x=0.5, conic_y=-0.5
        )
        assert_allclose(geom.sag(0, 0), 0.0)

    def test_sag_finite_radii(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=10.0, radius_y=20.0)
        assert_allclose(geom.sag(x=1, y=1), 0.07514126037252641)

    def test_sag_rx_infinite(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=be.inf, radius_y=20.0
        )  # Cylindrical along X
        # Sag should only depend on y. zx(any) = 0
        # zy(1) for Ry=20 is 0.02501563183003138
        assert_allclose(geom.sag(x=10, y=1), 0.02501563183003138)
        assert_allclose(geom.sag(x=-5, y=1), 0.02501563183003138)

    def test_sag_ry_infinite(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=be.inf
        )  # Cylindrical along Y
        # Sag should only depend on x. zy(any) = 0
        # zx(1) for Rx=10 is 0.05012562854249503
        assert_allclose(geom.sag(x=1, y=10), 0.05012562854249503)
        assert_allclose(geom.sag(x=1, y=-5), 0.05012562854249503)

    def test_sag_both_infinite_plane(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=be.inf, radius_y=be.inf)
        assert_allclose(geom.sag(x=10, y=20), 0.0)
        # Also test with conics, should still be zero
        geom_conic = BiconicGeometry(
            cs, radius_x=be.inf, radius_y=be.inf, conic_x=0.5, conic_y=-1.0
        )
        assert_allclose(geom_conic.sag(x=10, y=20), 0.0)

    def test_sag_with_conics(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=20.0, conic_x=-1.0, conic_y=0.5
        )
        expected_sag = 0.05 + 0.02502345130203264
        assert_allclose(geom.sag(x=1, y=1), expected_sag)

    def test_sag_array_input(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=10.0, radius_y=20.0)
        x = be.array([0, 1, 2])
        y = be.array([0, 1, 1])
        expected_sags = be.array([0.0, 0.07514126037252641, 0.22705672190612818])
        assert_allclose(geom.sag(x, y), expected_sags)

    def test_surface_normal_vertex(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=20.0, conic_x=0.5, conic_y=-0.5
        )
        nx, ny, nz = geom._surface_normal(x=0, y=0)
        assert_allclose(nx, 0.0)
        assert_allclose(ny, 0.0)
        assert_allclose(nz, -1.0)

    def test_surface_normal_spherical_case(self, set_test_backend):
        cs = CoordinateSystem()
        R = 10.0
        geom = BiconicGeometry(cs, radius_x=R, radius_y=R, conic_x=0.0, conic_y=0.0)
        nx, ny, nz = geom._surface_normal(x=1, y=1)
        assert_allclose(nx, 0.09950371902099892)
        assert_allclose(ny, 0.09950371902099892)
        assert_allclose(nz, -0.9900493732390136)

    def test_surface_normal_cylindrical_rx_inf(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=be.inf, radius_y=10.0, conic_y=0.0
        )  # Cylinder along X
        nx, ny, nz = geom._surface_normal(x=5, y=1)  # x shouldn't matter for dfdx=0
        assert_allclose(nx, 0.0)
        assert_allclose(ny, 0.1)
        assert_allclose(nz, -0.99498743710662)

    def test_surface_normal_array_input(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=10.0, conic_x=0.0, conic_y=0.0
        )  # Spherical R=10
        x = be.array([0, 1])
        y = be.array([0, 1])

        # Point (0,0): nx=0, ny=0, nz=-1
        # Point (1,1): nx=0.099503719, ny=0.099503719, nz=-0.990049373 (from spherical test)
        expected_nx = be.array([0.0, 0.09950371902099892])
        expected_ny = be.array([0.0, 0.09950371902099892])
        expected_nz = be.array([-1.0, -0.9900493732390136])

        nx_calc, ny_calc, nz_calc = geom._surface_normal(x, y)
        assert_allclose(nx_calc, expected_nx)
        assert_allclose(ny_calc, expected_ny)
        assert_allclose(nz_calc, expected_nz)

    def test_distance_simple(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=10.0, radius_y=20.0)
        rays = RealRays(
            x=0.0, y=0.0, z=-5.0, L=0.0, M=0.0, N=1.0, wavelength=0.55, intensity=1.0
        )
        # Sag at (0,0) is 0, so distance to surface at (0,0) should be 5.0
        assert_allclose(geom.distance(rays), 5.0, atol=1e-9)  # Newton-Raphson tolerance

    def test_distance_planar_biconic(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=be.inf, radius_y=be.inf, conic_x=0.0, conic_y=0.0
        )  # Planar
        rays = RealRays(
            x=1.0, y=1.0, z=-5.0, L=0.0, M=0.0, N=1.0, wavelength=0.55, intensity=1.0
        )
        # Distance to plane z=0 should be 5.0
        # The parent NewtonRaphsonGeometry's _intersection calls _intersection_plane if self.radius is inf.
        # In Biconic.__init__, self.radius is set to radius_x. So this path is tested.
        assert_allclose(geom.distance(rays), 5.0, atol=1e-9)

    def test_to_dict_from_dict(self, set_test_backend):
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
        assert geom_dict["tol"] == 1e-9
        assert geom_dict["max_iter"] == 50
        assert "radius" not in geom_dict
        assert "conic" not in geom_dict

        reconstructed_geom = BiconicGeometry.from_dict(geom_dict)
        assert isinstance(reconstructed_geom, BiconicGeometry)
        assert_allclose(reconstructed_geom.Rx, original_geom.Rx)
        assert_allclose(reconstructed_geom.Ry, original_geom.Ry)
        assert_allclose(reconstructed_geom.kx, original_geom.kx)
        assert_allclose(reconstructed_geom.ky, original_geom.ky)
        assert reconstructed_geom.tol == original_geom.tol
        assert reconstructed_geom.max_iter == original_geom.max_iter
        assert reconstructed_geom.cs.to_dict() == original_geom.cs.to_dict()

        # Check if the reconstructed dict is identical (it should be)
        assert reconstructed_geom.to_dict() == geom_dict

    def test_from_dict_missing_keys(self, set_test_backend):
        cs = CoordinateSystem()
        minimal_valid_dict = {
            "type": "BiconicGeometry",
            "cs": cs.to_dict(),
            "radius_x": 10.0,
            "radius_y": 20.0,
            # conic_x, conic_y, tol, max_iter will use defaults
        }

        # Test missing radius_x
        invalid_dict_rx = minimal_valid_dict.copy()
        del invalid_dict_rx["radius_x"]
        with pytest.raises(
            ValueError, match="Missing required BiconicGeometry keys: {'radius_x'}"
        ):
            BiconicGeometry.from_dict(invalid_dict_rx)

        # Test missing radius_y
        invalid_dict_ry = minimal_valid_dict.copy()
        del invalid_dict_ry["radius_y"]
        with pytest.raises(
            ValueError, match="Missing required BiconicGeometry keys: {'radius_y'}"
        ):
            BiconicGeometry.from_dict(invalid_dict_ry)

        # Test missing cs
        invalid_dict_cs = minimal_valid_dict.copy()
        del invalid_dict_cs["cs"]
        with pytest.raises(
            ValueError, match="Missing required BiconicGeometry keys: {'cs'}"
        ):
            BiconicGeometry.from_dict(invalid_dict_cs)

    def test_from_dict_default_conics_tol_max_iter(self, set_test_backend):
        cs = CoordinateSystem()
        geom_data = {
            "type": "BiconicGeometry",
            "cs": cs.to_dict(),
            "radius_x": 10.0,
            "radius_y": 20.0,
        }
        geom = BiconicGeometry.from_dict(geom_data)
        assert geom.Rx == 10.0
        assert geom.Ry == 20.0
        assert geom.kx == 0.0  # Default
        assert geom.ky == 0.0  # Default
        assert (
            geom.tol == 1e-10
        )  # Default from NewtonRaphsonGeometry via BiconicGeometry
        assert (
            geom.max_iter == 100
        )  # Default from NewtonRaphsonGeometry via BiconicGeometry
