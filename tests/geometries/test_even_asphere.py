# tests/geometries/test_even_asphere.py
"""
Tests for the EvenAsphere geometry class in optiland.geometries.
"""
import numpy as np

import optiland.backend as be
from optiland import geometries
from optiland.coordinate_system import CoordinateSystem
from optiland.rays import RealRays
from ..utils import assert_allclose


class TestEvenAsphere:
    """
    Tests for the EvenAsphere geometry, which represents a surface with an
    even-order aspheric departure from a base conic.
    """

    def test_str(self, set_test_backend):
        """
        Tests the string representation of the EvenAsphere geometry.
        """
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2],
        )
        assert str(geometry) == "Even Asphere"

    def test_sag(self, set_test_backend):
        """
        Tests the sag calculation for an even aspheric surface.
        """
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=27.0,
            conic=0.0,
            coefficients=[1e-3, -1e-5],
        )

        # Test sag at the origin
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at various points
        assert_allclose(geometry.sag(1, 1), 0.039022474574473776)
        assert_allclose(geometry.sag(-2, 3), 0.25313367948069593)

        # Test with array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([0.0, 1.1206923060227627, 1.3196652673420655])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        """
        Tests the calculation of the distance from a ray's origin to its
        intersection point with the surface.
        """
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

        # Test distance for a ray not parallel to the z-axis
        L = 0.222
        M = -0.229
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 10.625463223037386)

    def test_surface_normal(self, set_test_backend):
        """
        Tests the calculation of the surface normal vector at a given point.
        """
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
        """
        Tests the serialization of an EvenAsphere instance to a dictionary.
        """
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
        """
        Tests the deserialization of an EvenAsphere instance from a dictionary.
        """
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