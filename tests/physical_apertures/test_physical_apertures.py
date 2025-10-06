# tests/physical_apertures/test_physical_apertures.py
"""
Tests for the physical aperture classes in optiland.physical_apertures.

This file verifies that the various aperture shapes (Radial, Rectangular,
Elliptical) correctly determine if a point is inside the aperture and that
they can be correctly serialized and deserialized.
"""
import pytest
import optiland.backend as be
from optiland.physical_apertures.radial import RadialAperture
from optiland.physical_apertures.rectangular import RectangularAperture
from optiland.physical_apertures.elliptical import EllipticalAperture
from ..utils import assert_array_equal


class TestRadialAperture:
    """
    Tests the RadialAperture class.
    """

    def test_is_inside(self, set_test_backend):
        """
        Tests that the `is_inside` method correctly identifies points that
        are inside, on the boundary, and outside of the radial aperture.
        """
        aperture = RadialAperture(r_max=10.0)
        x = be.array([0, 5, 10, 15])
        y = be.array([0, 5, 0, 0])
        inside = aperture.is_inside(x, y)
        assert_array_equal(inside, be.asarray([True, True, True, False]))

    def test_to_dict(self, set_test_backend):
        """
        Tests the serialization of a RadialAperture instance to a dictionary.
        """
        aperture = RadialAperture(r_max=10.0)
        d = aperture.to_dict()
        assert d["type"] == "RadialAperture"
        assert d["r_max"] == 10.0

    def test_from_dict(self, set_test_backend):
        """
        Tests the deserialization of a RadialAperture instance from a
        dictionary.
        """
        d = {"type": "RadialAperture", "r_max": 10.0, "r_min": 0}
        aperture = RadialAperture.from_dict(d)
        assert aperture.r_max == 10.0


class TestRectangularAperture:
    """
    Tests the RectangularAperture class.
    """

    def test_is_inside(self, set_test_backend):
        """
        Tests that the `is_inside` method correctly identifies points inside
        and outside the rectangular aperture for both x and y boundaries.
        """
        aperture = RectangularAperture(x_min=-10, x_max=10, y_min=-5, y_max=5)
        # Test x-boundaries
        x_test = be.array([-15, -10, 0, 10, 15])
        y_test = be.zeros(5)
        inside_x = aperture.is_inside(x_test, y_test)
        assert_array_equal(inside_x, be.asarray([False, True, True, True, False]))

        # Test y-boundaries
        x_test = be.zeros(5)
        y_test = be.array([-10, -5, 0, 5, 10])
        inside_y = aperture.is_inside(x_test, y_test)
        assert_array_equal(inside_y, be.asarray([False, True, True, True, False]))

    def test_to_dict(self, set_test_backend):
        """
        Tests the serialization of a RectangularAperture instance to a dictionary.
        """
        aperture = RectangularAperture(x_min=-10, x_max=10, y_min=-5, y_max=5)
        d = aperture.to_dict()
        assert d["type"] == "RectangularAperture"
        assert d["x_min"] == -10
        assert d["x_max"] == 10
        assert d["y_min"] == -5
        assert d["y_max"] == 5

    def test_from_dict(self, set_test_backend):
        """
        Tests the deserialization of a RectangularAperture instance from a
        dictionary.
        """
        d = {"type": "RectangularAperture", "x_min": -10, "x_max": 10, "y_min": -5, "y_max": 5}
        aperture = RectangularAperture.from_dict(d)
        assert aperture.x_min == -10
        assert aperture.x_max == 10
        assert aperture.y_min == -5
        assert aperture.y_max == 5


class TestEllipticalAperture:
    """
    Tests the EllipticalAperture class.
    """

    def test_is_inside(self, set_test_backend):
        """
        Tests that the `is_inside` method correctly identifies points inside
        and outside the elliptical aperture for both x and y axes.
        """
        aperture = EllipticalAperture(x_radius=10, y_radius=5)
        # Test along x-axis
        x_test = be.array([0, 5, 10, 15])
        y_test = be.zeros(4)
        inside_x = aperture.is_inside(x_test, y_test)
        assert_array_equal(inside_x, be.asarray([True, True, True, False]))

        # Test along y-axis
        x_test = be.zeros(4)
        y_test = be.array([0, 2.5, 5, 7.5])
        inside_y = aperture.is_inside(x_test, y_test)
        assert_array_equal(inside_y, be.asarray([True, True, True, False]))

    def test_to_dict(self, set_test_backend):
        """
        Tests the serialization of an EllipticalAperture instance to a dictionary.
        """
        aperture = EllipticalAperture(x_radius=10, y_radius=5)
        d = aperture.to_dict()
        assert d["type"] == "EllipticalAperture"
        assert d["x_radius"] == 10
        assert d["y_radius"] == 5

    def test_from_dict(self, set_test_backend):
        """
        Tests the deserialization of an EllipticalAperture instance from a
        dictionary.
        """
        d = {"type": "EllipticalAperture", "x_radius": 10, "y_radius": 5}
        aperture = EllipticalAperture.from_dict(d)
        assert aperture.x_radius == 10
        assert aperture.y_radius == 5