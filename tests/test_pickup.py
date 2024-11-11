import pytest
from optiland.samples.objectives import CookeTriplet
from optiland.pickup import Pickup


class TestPickup:

    def test_apply_radius(self):
        lens = CookeTriplet()
        pickup = Pickup(lens, 1, 'radius', 2, scale=2, offset=3)
        pickup.apply()
        r0 = lens.surface_group.radii[1]
        r1 = lens.surface_group.radii[2]
        assert r1 == 2 * r0 + 3

    def test_apply_conic(self):
        lens = CookeTriplet()
        pickup = Pickup(lens, 1, 'conic', 2, scale=2, offset=3)
        pickup.apply()
        k0 = lens.surface_group.conic[1]
        k1 = lens.surface_group.conic[2]
        assert k1 == 2 * k0 + 3

    def test_apply_thickness(self):
        lens = CookeTriplet()
        pickup = Pickup(lens, 1, 'thickness', 2, scale=1, offset=0)
        pickup.apply()
        t1 = lens.surface_group.get_thickness(1)
        t2 = lens.surface_group.get_thickness(2)
        assert t1 == t2

    def test_invalid_attr_type(self):
        lens = CookeTriplet()
        pickup = Pickup(lens, 1, 'invalid', 2, scale=1, offset=0)
        with pytest.raises(ValueError):
            pickup.apply()

    def test_invalid_set_attr(self):
        lens = CookeTriplet()
        pickup = Pickup(lens, 1, 'invalid', 2, scale=1, offset=0)
        with pytest.raises(ValueError):
            pickup._set_value(5.0)
