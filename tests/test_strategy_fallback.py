""" tests for fallback ray aiming strategy """
import unittest
from unittest.mock import MagicMock

import numpy as np

from optiland.aiming.strategies import (
    FallbackAimingStrategy,
    IterativeAimingStrategy,
    ParaxialAimingStrategy,
)
from optiland.rays.real_rays import RealRays
from optiland.optic.optic import Optic
import optiland.backend as be

class TestFallbackAimingStrategySimple(unittest.TestCase):
    def setUp(self):
        self.optic = MagicMock(spec=Optic)
        self.optic.surface_group = MagicMock()
        self.optic.surface_group.stop_index = 1
        self.optic.surface_group.x = be.array([[0.0], [0.0]])
        self.optic.surface_group.y = be.array([[0.0], [0.0]])
        self.optic.surface_group.surfaces = [MagicMock(), MagicMock()]
        self.optic.surface_group.surfaces[1].aperture.r_max = 1.0
        self.primary_strategy = MagicMock(spec=IterativeAimingStrategy)
        self.secondary_strategy = MagicMock(spec=ParaxialAimingStrategy)

        self.fallback_strategy = FallbackAimingStrategy(
            primary=self.primary_strategy,
            secondary=self.secondary_strategy,
        )

        self.Hx, self.Hy = 0.0, 1.0
        self.Px, self.Py = 0.0, 0.0
        self.wavelength = 0.55

        self.mock_rays = RealRays(
            x=be.array([0.0]), y=be.array([0.0]), z=be.array([0.0]),
            L=be.array([0.0]), M=be.array([0.0]), N=be.array([1.0]),
            intensity=be.array([1.0]), wavelength=be.array([self.wavelength])
        )
        self.mock_rays.fail = be.array([False])

    def test_primary_succeeds(self):
        """Test that the primary strategy is used when it succeeds."""
        self.primary_strategy.aim_ray.return_value = self.mock_rays

        def trace_effect(rays):
            rays.fail = be.array([False])

        self.optic.surface_group.trace.side_effect = trace_effect

        result_rays = self.fallback_strategy.aim_ray(
            self.optic, self.Hx, self.Hy, self.Px, self.Py, self.wavelength
        )

        self.primary_strategy.aim_ray.assert_called_once()
        self.secondary_strategy.aim_ray.assert_not_called()
        self.assertIs(result_rays, self.mock_rays)

    def test_fallback_on_exception(self):
        """Test fallback to secondary strategy when primary raises an exception."""
        self.primary_strategy.aim_ray.side_effect = Exception("Primary failed")
        self.secondary_strategy.aim_ray.return_value = self.mock_rays

        result_rays = self.fallback_strategy.aim_ray(
            self.optic, self.Hx, self.Hy, self.Px, self.Py, self.wavelength
        )

        self.primary_strategy.aim_ray.assert_called_once()
        self.secondary_strategy.aim_ray.assert_called_once()
        self.assertIs(result_rays, self.mock_rays)

    def test_fallback_on_ray_fail(self):
        """Test fallback to secondary strategy when a ray fails to trace."""
        self.primary_strategy.aim_ray.return_value = self.mock_rays

        def trace_effect(rays):
            rays.fail = be.array([True])

        self.optic.surface_group.trace.side_effect = trace_effect
        self.secondary_strategy.aim_ray.return_value = self.mock_rays

        result_rays = self.fallback_strategy.aim_ray(
            self.optic, self.Hx, self.Hy, self.Px, self.Py, self.wavelength
        )

        self.primary_strategy.aim_ray.assert_called_once()
        self.secondary_strategy.aim_ray.assert_called_once()
        self.assertIs(result_rays, self.mock_rays)

if __name__ == "__main__":
    unittest.main()
