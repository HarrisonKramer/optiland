import unittest
from unittest.mock import Mock

from optiland.rays.aiming.context import RayAimingContext
from optiland.rays.aiming.strategy import RayAimingStrategy
from optiland.optic.optic import Optic


class DummyRayAimingStrategy(RayAimingStrategy):
    def aim_ray(self, optic: Optic, Hx: float, Hy: float, Px: float, Py: float, wavelength: float):
        return (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


class TestRayAimingContext(unittest.TestCase):
    def test_aim_ray_delegation(self):
        # GIVEN
        dummy_strategy = DummyRayAimingStrategy()
        dummy_strategy.aim_ray = Mock(return_value=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        context = RayAimingContext(strategy=dummy_strategy)
        optic = Mock(spec=Optic)

        # WHEN
        result = context.aim_ray(optic, 0.1, 0.2, 0.3, 0.4, 0.5)

        # THEN
        self.assertEqual(result, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        dummy_strategy.aim_ray.assert_called_once_with(optic, 0.1, 0.2, 0.3, 0.4, 0.5)

    def test_set_strategy(self):
        # GIVEN
        strategy1 = DummyRayAimingStrategy()
        strategy1.aim_ray = Mock(return_value=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        strategy2 = DummyRayAimingStrategy()
        strategy2.aim_ray = Mock(return_value=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0))

        context = RayAimingContext(strategy=strategy1)
        optic = Mock(spec=Optic)

        # WHEN
        context.set_strategy(strategy2)
        result = context.aim_ray(optic, 0.1, 0.2, 0.3, 0.4, 0.5)

        # THEN
        self.assertEqual(result, (10.0, 20.0, 30.0, 40.0, 50.0, 60.0))
        strategy1.aim_ray.assert_not_called()
        strategy2.aim_ray.assert_called_once_with(optic, 0.1, 0.2, 0.3, 0.4, 0.5)

    def test_aim_ray_without_strategy(self):
        # GIVEN
        context = RayAimingContext()
        optic = Mock(spec=Optic)

        # WHEN/THEN
        with self.assertRaises(ValueError):
            context.aim_ray(optic, 0.1, 0.2, 0.3, 0.4, 0.5)
