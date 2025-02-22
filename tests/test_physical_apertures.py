import numpy as np
from optiland.physical_apertures import (
    RadialAperture,
    OffsetRadialAperture,
    UnionAperture,
    IntersectionAperture,
    DifferenceAperture,
    RectangularAperture,
    EllipticalAperture,
    PolygonAperture
)
from optiland.rays import RealRays


class TestRadialAperture:
    def test_clip(self):
        aperture = RadialAperture(r_max=5, r_min=2)
        rays = RealRays([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5],
                        [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1])
        aperture.clip(rays)
        assert np.all(rays.i == [0, 0, 1, 1, 0, 0])

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
        assert aperture.to_dict() == {
            'type': 'RadialAperture',
            'r_max': 5,
            'r_min': 2
        }

    def test_from_dict(self):
        data = {
            'type': 'RadialAperture',
            'r_max': 5,
            'r_min': 2
        }
        aperture = RadialAperture.from_dict(data)
        assert aperture.r_max == 5
        assert aperture.r_min == 2
        assert isinstance(aperture, RadialAperture)


class TestOffsetRadialAperture:
    def test_clip(self):
        aperture = OffsetRadialAperture(
            r_max=5, r_min=2, offset_x=1, offset_y=1
        )
        rays = RealRays([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5],
                        [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1])
        aperture.clip(rays)
        assert np.all(rays.i == [0, 0, 0, 1, 1, 0])

    def test_scale(self):
        aperture = OffsetRadialAperture(
            r_max=5, r_min=2, offset_x=1, offset_y=1
        )
        aperture.scale(2)
        assert aperture.r_max == 10
        assert aperture.r_min == 4
        assert aperture.offset_x == 2
        assert aperture.offset_y == 2

        aperture = OffsetRadialAperture(
            r_max=5, r_min=2, offset_x=1, offset_y=1
        )
        aperture.scale(0.5)
        assert aperture.r_max == 2.5
        assert aperture.r_min == 1
        assert aperture.offset_x == 0.5
        assert aperture.offset_y == 0.5

    def test_to_dict(self):
        aperture = OffsetRadialAperture(
            r_max=5, r_min=2, offset_x=1, offset_y=1
        )
        assert aperture.to_dict() == {
            'type': 'OffsetRadialAperture',
            'r_max': 5,
            'r_min': 2,
            'offset_x': 1,
            'offset_y': 1
        }

    def test_from_dict(self):
        data = {
            'type': 'OffsetRadialAperture',
            'r_max': 5,
            'r_min': 2,
            'offset_x': 1,
            'offset_y': 1
        }
        aperture = OffsetRadialAperture.from_dict(data)
        assert aperture.r_max == 5
        assert aperture.r_min == 2
        assert aperture.offset_x == 1
        assert aperture.offset_y == 1
        assert isinstance(aperture, OffsetRadialAperture)


class TestBooleanApertures:
    def setup_method(self):
        self.aperture1 = RadialAperture(r_max=1)
        self.aperture2 = RadialAperture(r_max=0.5)

    def test_union_aperture(self):
        union_aperture = UnionAperture(self.aperture1, self.aperture2)
        x = np.array([0, 0.0, 0.7, 1.5])
        y = np.array([0, 0.5, 0.0, 1.5])
        result = union_aperture.contains(x, y)
        expected = np.array([True, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_intersection_aperture(self):
        intersection_aperture = IntersectionAperture(self.aperture1,
                                                     self.aperture2)
        x = np.array([0, 0.0, 0.7, 1.5])
        y = np.array([0, 0.5, 0.0, 1.5])
        result = intersection_aperture.contains(x, y)
        expected = np.array([True, True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_difference_aperture(self):
        difference_aperture = DifferenceAperture(self.aperture1,
                                                 self.aperture2)
        x = np.array([0, 0.0, 0.7, 1.5])
        y = np.array([0, 0.5, 0.0, 1.5])
        result = difference_aperture.contains(x, y)
        expected = np.array([False, False, True, False])
        np.testing.assert_array_equal(result, expected)
