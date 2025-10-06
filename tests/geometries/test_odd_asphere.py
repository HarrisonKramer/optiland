# tests/geometries/test_odd_asphere.py
"""
Tests for the OddAsphere geometry class in optiland.geometries.
"""
import optiland.backend as be
from optiland import geometries
from optiland.coordinate_system import CoordinateSystem
from optiland.rays import RealRays
from ..utils import assert_allclose


class TestOddAsphere:
    """
    Tests for the OddAsphere geometry, which represents a surface with an
    odd-order aspheric departure from a base conic.
    """

    def test_str(self, set_test_backend):
        """
        Tests the string representation of the OddAsphere geometry.
        """
        cs = CoordinateSystem()
        geometry = geometries.OddAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2, -1e-5],
        )
        assert str(geometry) == "Odd Asphere"

    def test_sag(self, set_test_backend):
        """
        Tests the sag calculation for an odd aspheric surface.
        """
        cs = CoordinateSystem()
        geometry = geometries.OddAsphere(
            cs,
            radius=27.0,
            conic=0.0,
            coefficients=[1e-3, -1e-5],
        )

        # Test sag at the origin
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at various points
        assert_allclose(geometry.sag(1, 1), 0.03845668813684687)
        assert_allclose(geometry.sag(-2, 3), 0.24529923075615997)

        # Test with array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([0.0, 1.10336808, 1.30564148])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        """
        Tests the calculation of the distance from a ray's origin to its
        intersection point with the surface.
        """
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
        """
        Tests the calculation of the surface normal vector at a given point.
        """
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