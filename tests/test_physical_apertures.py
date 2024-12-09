import numpy as np
from optiland import physical_apertures
from optiland.rays import RealRays


class TestRadialAperture:
    def test_clip(self):
        aperture = physical_apertures.RadialAperture(r_max=5, r_min=2)
        rays = RealRays([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5],
                        [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1])
        aperture.clip(rays)
        assert np.all(rays.i == [0, 0, 1, 1, 0, 0])

    def test_scale(self):
        aperture = physical_apertures.RadialAperture(r_max=5, r_min=2)
        aperture.scale(2)
        assert aperture.r_max == 10
        assert aperture.r_min == 4

        aperture = physical_apertures.RadialAperture(r_max=5, r_min=2)
        aperture.scale(0.5)
        assert aperture.r_max == 2.5
        assert aperture.r_min == 1

    def test_to_dict(self):
        aperture = physical_apertures.RadialAperture(r_max=5, r_min=2)
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
        aperture = physical_apertures.RadialAperture.from_dict(data)
        assert aperture.r_max == 5
        assert aperture.r_min == 2
        assert isinstance(aperture, physical_apertures.RadialAperture)
