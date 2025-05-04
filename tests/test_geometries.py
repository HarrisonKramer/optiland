import optiland.backend as be
import pytest
import numpy as np
from optiland.optic import Optic
from optiland.materials import Material, IdealMaterial
from optiland import geometries
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
        coefficients = be.array([
            [0.0, 1e-2, -2e-3],
            [0.1, 1e-2, -1e-3],
            [0.2, 1e-2, 0.0]
        ])
        geometry = geometries.PolynomialGeometry(
            cs,
            radius=22.0,
            conic=0.0,
            coefficients=coefficients,
        )
        assert str(geometry) == "Polynomial XY"

    def test_sag(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array([
            [0.0, 1e-2, -2e-3],
            [0.1, 1e-2, -1e-3],
            [0.2, 1e-2, 0.0]
        ])
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
        coefficients = be.array([
            [0.0, 1e-2, 2e-3],
            [0.1, -1e-2, 1e-3],
            [0.2, 1e-2, 2e-4]
        ])
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
        coefficients = be.array([
            [0.0, 1e-2, 2e-3],
            [0.1, -1e-2, 1e-3],
            [0.2, 1e-2, 2e-4]
        ])
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
        coefficients = be.array([
             [0.0, 1e-2, 2e-3],
             [0.1, -1e-2, 1e-3],
             [0.2, 1e-2, 2e-4]
        ])
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
        coefficients = be.array([
            [0.0, 1e-2, 2e-3],
            [0.1, -1e-2, 1e-3],
            [0.2, 1e-2, 2e-4]
        ])
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
        coefficients = be.array([
            [0.0, 1e-2, -2e-3],
            [0.1, 1e-2, -1e-3],
            [0.2, 1e-2, 0.0]
        ])
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
        coefficients = be.array([
            [0.0, 1e-2, -2e-3],
            [0.1, 1e-2, -1e-3],
            [0.2, 1e-2, 0.0]
        ])
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
        coefficients = be.array([
            [0.0, 1e-2, -2e-3],
            [0.1, 1e-2, -1e-3],
            [0.2, 1e-2, 0.0]
        ])
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
        coefficients = be.array([
            [0.0, 1e-2, -2e-3],
            [0.1, 1e-2, -1e-3],
            [0.2, 1e-2, 0.0]
        ])
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
        coefficients = be.array([
            [0.0, 1e-2, -2e-3],
            [0.1, 1e-2, -1e-3],
            [0.2, 1e-2, 0.0]
        ])
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
        coefficients = be.array([
            [0.0, 1e-2, -2e-3],
            [0.1, 1e-2, -1e-3],
            [0.2, 1e-2, 0.0]
        ])
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
        coefficients = be.array([
            [0.0, 1e-2, -2e-3],
            [0.1, 1e-2, -1e-3],
            [0.2, 1e-2, 0.0]
        ])
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
        coefficients = be.array([
            [0.0, 1e-2, -2e-3],
            [0, 0, 0],
            [0, 0, 0]
        ])
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
def basic_toroid_geometry():
    """Provides a basic ToroidalGeometry instance for testing"""
    cs = CoordinateSystem(x=0, y=0, z=0)
    radius_rotation = 100.0  # R (X-Z radius)
    radius_yz = 50.0         # R_y (Y-Z radius)
    conic = -0.5             # k_yz (YZ conic)
    coeffs_poly_y = [1e-5]   
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_rotation=radius_rotation,
        radius_yz=radius_yz,
        conic=conic,
        coeffs_poly_y=coeffs_poly_y
    )

@pytest.fixture
def cylinder_x_geometry():
    """Provides a cylindrical geometry (flat in X). R_rot = inf"""
    cs = CoordinateSystem(x=0, y=0, z=0)
    radius_rotation = be.inf # Flat in X-Z plane
    radius_yz = -50.0      
    conic = 0.0
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_rotation=radius_rotation,
        radius_yz=radius_yz,
        conic=conic
    )

@pytest.fixture
def cylinder_y_geometry():
    """Provides a cylindrical geometry (flat in Y). R_yz = inf"""
    cs = CoordinateSystem(x=0, y=0, z=0)
    radius_rotation = 100.0 
    radius_yz = be.inf      # Flat in Y-Z plane
    conic = 0.0
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_rotation=radius_rotation,
        radius_yz=radius_yz,
        conic=conic
    )

class TestToroidalGeometry:
    
    def test_toroidal_str(self):
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

    def test_toroidal_sag_vertex(self, basic_toroid_geometry):
        """Test sag at the vertex (0, 0). Should be 0 by definition."""
        x = be.array([0.0])
        y = be.array([0.0])
        assert basic_toroid_geometry.sag(x, y) == pytest.approx(0.0)

    def test_toroidal_normal_vertex(self, basic_toroid_geometry):
        """" Test the normal vector at the vertex (0, 0). Should be (0, 0, -1) """
        x = be.array([0.0])
        y = be.array([0.0])
        nx, ny, nz = basic_toroid_geometry._surface_normal(x, y)
        assert nx == pytest.approx(0.0)
        assert ny == pytest.approx(0.0)
        assert nz == pytest.approx(-1.0)
        
    def test_toroidal_sag_known_points(self, basic_toroid_geometry):
        """Test sag at specific points"""
        x = be.array([0.0, 10.0, 0.0])
        y = be.array([10.0, 0.0, 5.0])

        expected_z_0_10 = 1.00605051
        expected_z_10_0 = 0.50125628
        expected_z_0_5 = 0.25056330

        expected_z = be.array([expected_z_0_10, expected_z_10_0, expected_z_0_5])  
        calculated_z = basic_toroid_geometry.sag(x, y)
        assert be.allclose(calculated_z, expected_z, rtol=1e-5, atol=1e-6)    
    
    def test_toroidal_normal_known_points(self, basic_toroid_geometry):
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
        
    def test_cylinder_x_sag(self, cylinder_x_geometry):
        """Test sag for cylinder flat in X. Should only depend on y."""
        x = be.array([0.0, 10.0, 10.0])
        y = be.array([5.0, 5.0, 0.0])
        
        expected_z = be.array([-0.25062818, -0.25062818, 0.0]) 
        calculated_z = cylinder_x_geometry.sag(x, y)
        be.testing.assert_allclose(calculated_z, expected_z, rtol=1e-5, atol=1e-6)    
        
    def test_cylinder_y_sag(self, cylinder_y_geometry):
        """Test sag for cylinder flat in Y. Should only depend on x."""
        x = be.array([5.0, 5.0, 0.0])
        y = be.array([0.0, 10.0, 10.0])
        
        expected_z = be.array([0.12507822, 0.12507822, 0.0]) 
        calculated_z = cylinder_y_geometry.sag(x, y)
        be.testing.assert_allclose(calculated_z, expected_z, rtol=1e-5, atol=1e-6)
    
    def test_toroidal_sag_vs_zemax(self, basic_toroid_geometry):
        """
        Compares sag values calculated by Optiland with
        Zemax data for the basic_toroid_geometry.
        """
        geometry = basic_toroid_geometry

        x_coords = be.array([0.0,  2.5, 0.0, -2.5,  5.0,  -5.0,  2.5, -2.5])
        y_coords = be.array([0.0,  0.0, 2.5,  0.0,  2.5,  -2.5, -2.5,  2.5])

        # --- Zemax Sag Data ---
        zemax_z_sag = be.array([
            0.0,                      # (0, 0) - Vertex
            3.125488433897521E-002,   # (2.5, 0) 
            6.258204346657634E-002,   # (0, 2.5) 
            3.125488433897521E-002,   # (-2.5, 0) 
            1.877386899843393E-001,   # (5, 2.5) 
            1.877386899843393E-001,   # (-5, -2.5) 
            9.385650612446353E-002,   # (2.5, -2.5) 
            9.385650612446353E-002    # (-2.5, 2.5) 
        ]) 

        # Calculate sag using Optiland
        optiland_z_sag = geometry.sag(x_coords, y_coords)

        assert be.allclose(optiland_z_sag,zemax_z_sag,rtol=1e-5,atol=1e-6)

    
    def test_toroidal_ray_tracing_comparison(self):
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

        num_rays = 5 # Number of rays per fan
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
        rays_in_yfan = RealRays(x=x_in_yfan, y=y_in_yfan, z=z_in_yfan,
                                L=L_in_yfan, M=M_in_yfan, N=N_in_yfan,
                                wavelength=wavelength, intensity=intensity_yfan)

        # Trace Y-Fan Rays
        rays_out_yfan = lens.surface_group.trace(rays_in_yfan)
        print("Y-Fan Rays:")
        print(rays_out_yfan.y)
        zemax_x_out_yfan = be.array([0.0] * num_rays) 
        zemax_y_out_yfan = be.array([-8.123193233401276E-001, -4.676255499616224E-001, 0, 4.676255499616224E-001, 8.123193233401276E-001]) 
        zemax_z_out_yfan = be.array([15.0] * num_rays) 
        zemax_L_out_yfan = be.array([0.0] * num_rays) 
        zemax_M_out_yfan = be.array([3.251509839270260E-001, 1.537950377308984E-001, 0.0, -1.537950377308984E-001, -3.251509839270260E-001]) 
        zemax_N_out_yfan = be.array([9.456621160072382E-001, 9.881027711576116E-001, 1.0, 9.881027711576116E-001, 9.456621160072382E-001]) 

        # Comparison Assertions for Y-Fan
        assert be.allclose(rays_out_yfan.x, zemax_x_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.y, zemax_y_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.z, zemax_z_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.L, zemax_L_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.M, zemax_M_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.N, zemax_N_out_yfan, rtol=1e-5, atol=1e-6)

        

    def test_toroidal_to_dict(self, basic_toroid_geometry):
        """Test serialization to dictionary."""
        geom_dict = basic_toroid_geometry.to_dict()
        assert geom_dict["type"] == "ToroidalGeometry" 
        assert geom_dict["geometry_type"] == "Toroidal" 
        assert geom_dict["radius_rotation"] == 100.0
        assert geom_dict["radius_yz"] == 50.0
        assert geom_dict["conic_yz"] == -0.5
        assert geom_dict["coeffs_poly_y"] == [1e-5]
        
        assert "radius" not in geom_dict
        assert "conic" not in geom_dict
        assert "coefficients" not in geom_dict

    def test_toroidal_from_dict(self, basic_toroid_geometry):
        """Test deserialization from dictionary."""
        geom_dict = basic_toroid_geometry.to_dict()
        new_geometry = geometries.ToroidalGeometry.from_dict(geom_dict) 
        assert isinstance(new_geometry, geometries.ToroidalGeometry)
        assert new_geometry.to_dict() == geom_dict 

    def test_toroidal_from_dict_invalid(self):
        """Test deserialization with missing keys."""
        cs = CoordinateSystem()
        invalid_dict = {
            "type": "ToroidalGeometry",
            "cs": cs.to_dict(),
            # Missing radius_rotation, radius_yz
        }
        with pytest.raises(ValueError):
            geometries.ToroidalGeometry.from_dict(invalid_dict)
            
            