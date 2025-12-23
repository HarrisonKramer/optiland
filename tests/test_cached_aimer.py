
import unittest
from unittest.mock import MagicMock, call
import optiland.backend as be
from optiland.rays.ray_aiming.cached import CachedRayAimer
from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.optic import Optic

class TestCachedRayAimer(unittest.TestCase):
    def setUp(self):
        # Create a mock optic
        self.optic = MagicMock(spec=Optic)
        surface_group_mock = MagicMock()
        surface_group_mock.to_dict.return_value = {"surfaces": []}
        self.optic.surface_group = surface_group_mock
        
        # Create a mock wrapped aimer
        self.wrapped_aimer = MagicMock(spec=BaseRayAimer)
        
        # Instantiate CachedRayAimer
        self.cached_aimer = CachedRayAimer(self.optic, self.wrapped_aimer)

    def test_cache_miss_calls_wrapped(self):
        """Test that the wrapped aimer is called on a cache miss."""
        inputs = ((0.0,), 0.55, (0.0, 0.0))
        expected_result = (1, 2, 3, 4, 5, 6)
        self.wrapped_aimer.aim_rays.return_value = expected_result
        
        result = self.cached_aimer.aim_rays(*inputs)
        
        self.assertEqual(result, expected_result)
        self.wrapped_aimer.aim_rays.assert_called_once()

    def test_cache_hit_skips_wrapped(self):
        """Test that the wrapped aimer is NOT called on a cache hit."""
        inputs = ((0.0,), 0.55, (0.0, 0.0))
        expected_result = (1, 2, 3, 4, 5, 6)
        self.wrapped_aimer.aim_rays.return_value = expected_result
        
        # First call (miss)
        self.cached_aimer.aim_rays(*inputs)
        self.wrapped_aimer.aim_rays.reset_mock()
        
        # Second call (hit)
        result = self.cached_aimer.aim_rays(*inputs)
        
        self.assertEqual(result, expected_result)
        self.wrapped_aimer.aim_rays.assert_not_called()

    def test_system_change_uses_cached_as_guess(self):
        """Test that a system change uses the cached result as a starting guess."""
        inputs = ((0.0,), 0.55, (0.0, 0.0))
        result1 = (1, 2, 3, 4, 5, 6)
        self.wrapped_aimer.aim_rays.return_value = result1
        
        # First call (miss)
        self.cached_aimer.aim_rays(*inputs)
        
        # Perturb system (change dict representation)
        self.optic.surface_group.to_dict.return_value = {"surfaces": ["changed"]}
        
        # Second call
        result2 = (7, 8, 9, 10, 11, 12)
        self.wrapped_aimer.aim_rays.return_value = result2
        
        final_result = self.cached_aimer.aim_rays(*inputs)
        
        self.assertEqual(final_result, result2)
        # Check that wrapped aimer was called with initial_guess=result1
        self.wrapped_aimer.aim_rays.assert_called_with(
            inputs[0], inputs[1], inputs[2], initial_guess=result1
        )

    def test_different_inputs_miss_cache(self):
        """Test that different inputs do not trigger a cache hit."""
        inputs1 = ((0.0,), 0.55, (0.0, 0.0))
        inputs2 = ((0.1,), 0.55, (0.0, 0.0))
        
        self.cached_aimer.aim_rays(*inputs1)
        self.wrapped_aimer.aim_rays.reset_mock()
        
        self.cached_aimer.aim_rays(*inputs2)
        self.wrapped_aimer.aim_rays.assert_called_once()

    def test_clear_cache(self):
        """Test clearing the cache."""
        inputs = ((0.0,), 0.55, (0.0, 0.0))
        self.cached_aimer.aim_rays(*inputs)
        self.wrapped_aimer.aim_rays.reset_mock()
        
        self.cached_aimer.clear_cache()
        self.cached_aimer.aim_rays(*inputs)
        
        self.wrapped_aimer.aim_rays.assert_called_once()

    def test_max_cache_size(self):
        """Test that the cache does not exceed max size."""
        self.cached_aimer.max_cache_size = 1
        inputs1 = ((0.0,), 0.55, (0.0, 0.0))
        inputs2 = ((0.1,), 0.55, (0.0, 0.0))
        
        self.cached_aimer.aim_rays(*inputs1)
        self.cached_aimer.aim_rays(*inputs2)
        
        # Cache should now only contain inputs2. inputs1 should be evicted.
        self.wrapped_aimer.aim_rays.reset_mock()
        
        # Call inputs1 again -> should be a miss (wrapped called)
        self.cached_aimer.aim_rays(*inputs1)
        self.wrapped_aimer.aim_rays.assert_called_once()

if __name__ == "__main__":
    unittest.main()
