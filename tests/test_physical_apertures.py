import tempfile
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import optiland.backend as be

from optiland.physical_apertures import (
    DifferenceAperture,
    EllipticalAperture,
    FileAperture,
    IntersectionAperture,
    OffsetRadialAperture,
    PolygonAperture,
    RadialAperture,
    RectangularAperture,
    UnionAperture,
)
from optiland.rays import RealRays

matplotlib.use("Agg")  # use non-interactive backend for testing


class TestRadialAperture:
    def test_clip(self):
        aperture = RadialAperture(r_max=5, r_min=2)
        rays = RealRays(
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        )
        aperture.clip(rays)
        assert be.all(rays.i == [0, 0, 1, 1, 0, 0])

    def test_scale(self):
        aperture = RadialAperture(r_max=5, r_min=2)
        aperture.scale(2)
        assert aperture.r_max == 10
        assert aperture.r_min == 4

        aperture = RadialAperture(r_max=5, r_min=2)
        aperture.scale(0.5)
        assert aperture.r_max == 2.5
        assert aperture.r_min == 1

    def test_to_dict(self):
        aperture = RadialAperture(r_max=5, r_min=2)
        assert aperture.to_dict() == {"type": "RadialAperture", "r_max": 5, "r_min": 2}

    def test_from_dict(self):
        data = {"type": "RadialAperture", "r_max": 5, "r_min": 2}
        aperture = RadialAperture.from_dict(data)
        assert aperture.r_max == 5
        assert aperture.r_min == 2
        assert isinstance(aperture, RadialAperture)

    @patch("matplotlib.pyplot.show")
    def test_view(self, mock_show):
        aperture = RadialAperture(r_max=5, r_min=2)
        aperture.view()
        plt.show()
        mock_show.assert_called_once()
        plt.close()

    def test_extent(self):
        aperture = RadialAperture(r_max=5, r_min=2)
        assert aperture.extent == (-5, 5, -5, 5)


class TestOffsetRadialAperture:
    def test_clip(self):
        aperture = OffsetRadialAperture(r_max=5, r_min=2, offset_x=1, offset_y=1)
        rays = RealRays(
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        )
        aperture.clip(rays)
        assert be.all(rays.i == [0, 0, 0, 1, 1, 0])

    def test_scale(self):
        aperture = OffsetRadialAperture(r_max=5, r_min=2, offset_x=1, offset_y=1)
        aperture.scale(2)
        assert aperture.r_max == 10
        assert aperture.r_min == 4
        assert aperture.offset_x == 2
        assert aperture.offset_y == 2

        aperture = OffsetRadialAperture(r_max=5, r_min=2, offset_x=1, offset_y=1)
        aperture.scale(0.5)
        assert aperture.r_max == 2.5
        assert aperture.r_min == 1
        assert aperture.offset_x == 0.5
        assert aperture.offset_y == 0.5

    def test_to_dict(self):
        aperture = OffsetRadialAperture(r_max=5, r_min=2, offset_x=1, offset_y=1)
        assert aperture.to_dict() == {
            "type": "OffsetRadialAperture",
            "r_max": 5,
            "r_min": 2,
            "offset_x": 1,
            "offset_y": 1,
        }

    def test_from_dict(self):
        data = {
            "type": "OffsetRadialAperture",
            "r_max": 5,
            "r_min": 2,
            "offset_x": 1,
            "offset_y": 1,
        }
        aperture = OffsetRadialAperture.from_dict(data)
        assert aperture.r_max == 5
        assert aperture.r_min == 2
        assert aperture.offset_x == 1
        assert aperture.offset_y == 1
        assert isinstance(aperture, OffsetRadialAperture)

    def test_extent(self):
        aperture = OffsetRadialAperture(r_max=5, r_min=2, offset_x=1, offset_y=1)
        assert aperture.extent == (-4, 6, -4, 6)


class TestBooleanApertures:
    def setup_method(self):
        self.aperture1 = RadialAperture(r_max=1)
        self.aperture2 = RadialAperture(r_max=0.5)

    def test_union_aperture(self):
        union_aperture = UnionAperture(self.aperture1, self.aperture2)
        x = be.array([0, 0.0, 0.7, 1.5])
        y = be.array([0, 0.5, 0.0, 1.5])
        result = union_aperture.contains(x, y)
        expected = be.array([True, True, True, False])
        be.testing.assert_array_equal(result, expected)

    def test_intersection_aperture(self):
        intersection_aperture = IntersectionAperture(self.aperture1, self.aperture2)
        x = be.array([0, 0.0, 0.7, 1.5])
        y = be.array([0, 0.5, 0.0, 1.5])
        result = intersection_aperture.contains(x, y)
        expected = be.array([True, True, False, False])
        be.testing.assert_array_equal(result, expected)

    def test_difference_aperture(self):
        difference_aperture = DifferenceAperture(self.aperture1, self.aperture2)
        x = be.array([0, 0.0, 0.7, 1.5])
        y = be.array([0, 0.5, 0.0, 1.5])
        result = difference_aperture.contains(x, y)
        expected = be.array([False, False, True, False])
        be.testing.assert_array_equal(result, expected)

    def test_union_type(self):
        a = self.aperture1 | self.aperture2
        assert isinstance(a, UnionAperture)

        a = self.aperture1 + self.aperture2
        assert isinstance(a, UnionAperture)

    def test_intersection_type(self):
        a = self.aperture1 & self.aperture2
        assert isinstance(a, IntersectionAperture)

    def test_difference_type(self):
        a = self.aperture1 - self.aperture2
        assert isinstance(a, DifferenceAperture)

    def test_extent(self):
        union_aperture = UnionAperture(self.aperture1, self.aperture2)
        assert union_aperture.extent == (-1, 1, -1, 1)

        intersection_aperture = IntersectionAperture(self.aperture1, self.aperture2)
        assert intersection_aperture.extent == (-1, 1, -1, 1)

        difference_aperture = DifferenceAperture(self.aperture1, self.aperture2)
        assert difference_aperture.extent == (-1, 1, -1, 1)


class TestRectangularAperture:
    def setup_method(self):
        self.aperture = RectangularAperture(x_min=-1, x_max=1, y_min=-0.5, y_max=0.5)

    def test_clip(self):
        rays = RealRays(
            [0, 1, 2, 3, 4, 5],
            [0, 0.5, 2, 3, 4, 5],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        )
        self.aperture.clip(rays)
        assert be.all(rays.i == [1, 1, 0, 0, 0, 0])

    def test_scale(self):
        self.aperture.scale(2)
        assert self.aperture.x_min == -2
        assert self.aperture.x_max == 2
        assert self.aperture.y_min == -1
        assert self.aperture.y_max == 1

        self.aperture.scale(0.5)
        assert self.aperture.x_min == -1
        assert self.aperture.x_max == 1
        assert self.aperture.y_min == -0.5
        assert self.aperture.y_max == 0.5

    def test_to_dict(self):
        assert self.aperture.to_dict() == {
            "type": "RectangularAperture",
            "x_min": -1,
            "x_max": 1,
            "y_min": -0.5,
            "y_max": 0.5,
        }

    def test_from_dict(self):
        data = {
            "type": "RectangularAperture",
            "x_min": -1,
            "x_max": 1,
            "y_min": -0.5,
            "y_max": 86,
        }
        aperture = RectangularAperture.from_dict(data)
        assert aperture.x_min == -1
        assert aperture.x_max == 1
        assert aperture.y_min == -0.5
        assert aperture.y_max == 86
        assert isinstance(aperture, RectangularAperture)

    def test_extent(self):
        assert self.aperture.extent == (-1, 1, -0.5, 0.5)


class TestEllipticalAperture:
    def setup_method(self):
        self.aperture = EllipticalAperture(a=1, b=0.5)

    def test_clip(self):
        rays = RealRays(
            [0, 1, 0, 3, 4, 5],
            [0, 0, 0.5, 3, 4, 5],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        )
        self.aperture.clip(rays)
        assert be.all(rays.i == [1, 1, 1, 0, 0, 0])

    def test_scale(self):
        self.aperture.scale(2)
        assert self.aperture.a == 2
        assert self.aperture.b == 1

        self.aperture.scale(0.5)
        assert self.aperture.a == 1
        assert self.aperture.b == 0.5

    def test_to_dict(self):
        assert self.aperture.to_dict() == {
            "type": "EllipticalAperture",
            "a": 1,
            "b": 0.5,
            "offset_x": 0,
            "offset_y": 0,
        }

    def test_from_dict(self):
        data = {
            "type": "EllipticalAperture",
            "a": 1,
            "b": 0.5,
            "offset_x": 0,
            "offset_y": 0.123,
        }
        aperture = EllipticalAperture.from_dict(data)
        assert aperture.a == 1
        assert aperture.b == 0.5
        assert aperture.offset_x == 0
        assert aperture.offset_y == 0.123
        assert isinstance(aperture, EllipticalAperture)

    def test_extent(self):
        assert self.aperture.extent == (-1, 1, -0.5, 0.5)


class TestPolygonAperture:
    def setup_method(self):
        self.aperture = PolygonAperture(x=[-10, 10, 10, -10], y=[-15, -15, 15, 15])

    def test_clip(self):
        rays = RealRays(
            [0, 5, 0, 10, 20, 20],
            [0, 0, 6, 15, 0, 21],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        )
        self.aperture.clip(rays)
        assert be.all(rays.i == [1, 1, 1, 1, 0, 0])

    def test_scale(self):
        self.aperture.scale(2)
        assert be.all(
            self.aperture.vertices
            == be.array([[-20, -30], [20, -30], [20, 30], [-20, 30]]),
        )

        self.aperture.scale(0.5)
        assert be.all(
            self.aperture.vertices
            == be.array([[-10, -15], [10, -15], [10, 15], [-10, 15]]),
        )

    def test_to_dict(self):
        data = self.aperture.to_dict()
        assert data["type"] == "PolygonAperture"
        assert be.all(data["x"] == [-10, 10, 10, -10])
        assert be.all(data["y"] == [-15, -15, 15, 15])

    def test_from_dict(self):
        data = {"type": "PolygonAperture", "x": [0, 1, 1, 0], "y": [0, 0, 1, 1]}
        aperture = PolygonAperture.from_dict(data)
        assert be.all(aperture.x == [0, 1, 1, 0])
        assert be.all(aperture.y == [0, 0, 1, 1])
        assert isinstance(aperture, PolygonAperture)

    def test_extent(self):
        assert self.aperture.extent == (-10, 10, -15, 15)


class TestFileAperture:
    def setup_method(self, temp_aperture_file):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("0, 0\n1, 0\n1, 1\n0, 1")
            temp_path = f.name
        self.aperture = FileAperture(temp_path, delimiter=",")

    def test_clip(self):
        rays = RealRays(
            [0.5, 0, 1, 10, 20, 20],
            [0.5, 1, 1, 15, 0, 21],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        )
        self.aperture.clip(rays)
        assert be.all(rays.i == [1, 1, 1, 0, 0, 0])

    def test_scale(self):
        self.aperture.scale(2)
        assert be.all(
            self.aperture.vertices == be.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
        )

    def test_to_dict(self):
        data = self.aperture.to_dict()
        assert data["type"] == "FileAperture"
        assert data["filepath"] == self.aperture.filepath
        assert data["delimiter"] == ","
        assert data["skip_header"] == 0
        assert be.all(data["x"] == [0, 1, 1, 0])
        assert be.all(data["y"] == [0, 0, 1, 1])

    def test_from_dict(self):
        data = {
            "type": "FileAperture",
            "filepath": self.aperture.filepath,
            "delimiter": ",",
            "skip_header": 0,
        }
        aperture = FileAperture.from_dict(data)
        assert aperture.filepath == self.aperture.filepath
        assert isinstance(aperture, FileAperture)

    def test_extent(self):
        assert self.aperture.extent == (0, 1, 0, 1)
