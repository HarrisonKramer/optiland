import unittest
from unittest.mock import Mock

from optiland.aiming import RayAiming
from optiland.aiming.strategies import RayAimingStrategy
from optiland.optic.optic import Optic


class DummyRayAimingStrategy(RayAimingStrategy):
    def aim(self, optic: Optic, Hx: float, Hy: float, Px: float, Py: float, wavelength: float):
        return (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


class TestRayAiming(unittest.TestCase):
    def test_aim_delegation(self):
        # GIVEN
        dummy_strategy = DummyRayAimingStrategy()
        dummy_strategy.aim = Mock(return_value=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        optic = Mock(spec=Optic)
        aiming = RayAiming(optic=optic, strategy=dummy_strategy)

        # WHEN
        result = aiming.aim(optic, 0.1, 0.2, 0.3, 0.4, 0.5)

        # THEN
        self.assertEqual(result, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        dummy_strategy.aim.assert_called_once_with(optic=optic, Hx=0.1, Hy=0.2, Px=0.3, Py=0.4, wavelength=0.5)

    def test_set_strategy(self):
        # GIVEN
        strategy1 = DummyRayAimingStrategy()
        strategy1.aim = Mock(return_value=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        strategy2 = DummyRayAimingStrategy()
        strategy2.aim = Mock(return_value=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0))
        optic = Mock(spec=Optic)
        aiming = RayAiming(optic=optic, strategy=strategy1)

        # WHEN
        aiming.set_strategy(strategy2)
        result = aiming.aim(optic, 0.1, 0.2, 0.3, 0.4, 0.5)

        # THEN
        self.assertEqual(result, (10.0, 20.0, 30.0, 40.0, 50.0, 60.0))
        strategy1.aim.assert_not_called()
        strategy2.aim.assert_called_once_with(optic=optic, Hx=0.1, Hy=0.2, Px=0.3, Py=0.4, wavelength=0.5)
