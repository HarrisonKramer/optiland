import unittest
from unittest.mock import Mock

from optiland.aiming import RayAiming
from optiland.aiming.strategies import RayAimingStrategy
from optiland.optic.optic import Optic


class DummyRayAimingStrategy(RayAimingStrategy):
    def aim_ray(self, optic: Optic, Hx: float, Hy: float, Px: float, Py: float, wavelength: float):
        return (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


class TestRayAiming(unittest.TestCase):
    def test_aim_ray_delegation(self):
        # GIVEN
        dummy_strategy = DummyRayAimingStrategy()
        dummy_strategy.aim_ray = Mock(return_value=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        aiming = RayAiming(strategy=dummy_strategy)
        optic = Mock(spec=Optic)

        # WHEN
        result = aiming.aim_ray(optic, 0.1, 0.2, 0.3, 0.4, 0.5)

        # THEN
        self.assertEqual(result, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        dummy_strategy.aim_ray.assert_called_once_with(optic, 0.1, 0.2, 0.3, 0.4, 0.5)

    def test_set_strategy(self):
        # GIVEN
        strategy1 = DummyRayAimingStrategy()
        strategy1.aim_ray = Mock(return_value=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        strategy2 = DummyRayAimingStrategy()
        strategy2.aim_ray = Mock(return_value=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0))

        aiming = RayAiming(strategy=strategy1)
        optic = Mock(spec=Optic)

        # WHEN
        aiming.set_strategy(strategy2)
        result = aiming.aim_ray(optic, 0.1, 0.2, 0.3, 0.4, 0.5)

        # THEN
        self.assertEqual(result, (10.0, 20.0, 30.0, 40.0, 50.0, 60.0))
        strategy1.aim_ray.assert_not_called()
        strategy2.aim_ray.assert_called_once_with(optic, 0.1, 0.2, 0.3, 0.4, 0.5)

    def test_aim_ray_without_strategy(self):
        # GIVEN
        aiming = RayAiming()
        optic = Mock(spec=Optic)

        # WHEN/THEN
        with self.assertRaises(ValueError):
            aiming.aim_ray(optic, 0.1, 0.2, 0.3, 0.4, 0.5)
