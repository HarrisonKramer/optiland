# tests/geometries/test_chebyshev.py
"""
Tests for the ChebyshevPolynomialGeometry class in optiland.geometries.
"""
import pytest
import numpy as np

import optiland.backend as be
from optiland import geometries
from optiland.coordinate_system import CoordinateSystem
from optiland.rays import RealRays
from ..utils import assert_allclose


class TestChebyshevGeometry:
    """
    Tests for the ChebyshevPolynomialGeometry class, which represents a
    surface with a departure defined by a 2D Chebyshev polynomial.
    """

    def test_str(self, set_test_backend):
        """
        Tests the string representation of the ChebyshevPolynomialGeometry.
        """
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
        """
        Tests the sag calculation for a Chebyshev polynomial surface.
        """
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

        # Test sag at the origin
        assert_allclose(geometry.sag(), -0.198)

        # Test sag at various points
        assert_allclose(geometry.sag(1, 1), -0.13832040010014895)
        assert_allclose(geometry.sag(-2, -7), 1.036336507973306)

        # Test with array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([-0.198, 1.22291856, 1.75689642])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        """
        Tests the calculation of the distance from a ray's origin to its
        intersection point with the surface.
        """
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

        # Test distance for a ray not parallel to the z-axis
        L = 0.164
        M = -0.210
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 10.29015593)

    def test_surface_normal(self, set_test_backend):
        """
        Tests the calculation of the surface normal vector at a given point.
        """
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
        """
        Tests that providing coordinates outside the normalization radius
        raises a ValueError.
        """
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
        """
        Tests the serialization of a ChebyshevPolynomialGeometry instance to a
        dictionary.
        """
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
        """
        Tests the deserialization of a ChebyshevPolynomialGeometry instance
        from a dictionary.
        """
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