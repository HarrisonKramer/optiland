# tests/geometries/test_polynomial.py
"""
Tests for the PolynomialGeometry class in optiland.geometries.
"""
import numpy as np

import optiland.backend as be
from optiland import geometries
from optiland.coordinate_system import CoordinateSystem
from optiland.rays import RealRays
from ..utils import assert_allclose


class TestPolynomialGeometry:
    """
    Tests for the PolynomialGeometry class, which represents a surface with a
    departure defined by a 2D polynomial in x and y.
    """

    def test_str(self, set_test_backend):
        """
        Tests the string representation of the PolynomialGeometry.
        """
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
        """
        Tests the sag calculation for a polynomial surface.
        """
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

        # Test sag at the origin
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at various points
        assert_allclose(geometry.sag(1, 1), 0.3725015998998511)
        assert_allclose(geometry.sag(-2, -7), 1.6294605079733058)

        # Test with array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([0.0, 2.305232559449707, 16.702875375272402])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        """
        Tests the calculation of the distance from a ray's origin to its
        intersection point with the surface.
        """
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

        # Test distance for a ray not parallel to the z-axis
        L = 0.164
        M = -0.210
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 12.610897321951025)

    def test_surface_normal(self, set_test_backend):
        """
        Tests the calculation of the surface normal vector at a given point.
        """
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
        """
        Tests the serialization of a PolynomialGeometry instance to a dictionary.
        """
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
        """
        Tests the deserialization of a PolynomialGeometry instance from a
        dictionary.
        """
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