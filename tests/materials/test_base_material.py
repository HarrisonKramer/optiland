# tests/materials/test_base.py
"""
Tests for the BaseMaterial class in optiland.materials.base.
"""
from unittest.mock import MagicMock

import optiland.backend as be
import numpy as np

from optiland.materials.base import BaseMaterial


class TestBaseMaterial:
    """
    Tests the caching mechanism and basic functionality of the BaseMaterial class.
    """
    def test_caching(self, set_test_backend):
        """
        Verifies that the caching mechanism for refractive index calculations
        works correctly for scalar values, numpy arrays, and torch tensors.
        """
        class DummyMaterial(BaseMaterial):
            """A dummy material for testing caching."""
            def _calculate_n(self, wavelength, **kwargs):
                pass

            def _calculate_k(self, wavelength, **kwargs):
                pass

        material = DummyMaterial()
        material._calculate_n = MagicMock(return_value=1.5)

        # Test with scalar value - should be a cache miss on first call
        result1 = material.n(0.5, temperature=25)
        assert result1 == 1.5
        material._calculate_n.assert_called_once_with(0.5, temperature=25)

        # Second call with same arguments should be a cache hit
        result2 = material.n(0.5, temperature=25)
        assert result2 == 1.5
        material._calculate_n.assert_called_once()

        # Test with numpy array - should be a cache miss
        wavelength_np = np.array([0.5, 0.6])
        material.n(wavelength_np, temperature=25)
        assert material._calculate_n.call_count == 2

        # Second call with same numpy array should be a cache hit
        material.n(wavelength_np, temperature=25)
        assert material._calculate_n.call_count == 2

        # Test with torch tensor if backend is torch
        if set_test_backend == "torch":
            # a torch tensor with same values should be a cache hit
            wavelength_torch = be.asarray(np.array([0.5, 0.6]))
            material.n(wavelength_torch, temperature=25)
            assert material._calculate_n.call_count == 2

            # a torch tensor with different values should be a cache miss
            wavelength_torch_2 = be.asarray(np.array([0.7, 0.8]))
            material.n(wavelength_torch_2, temperature=25)
            assert material._calculate_n.call_count == 3

            # and a cache hit
            material.n(wavelength_torch_2, temperature=25)
            assert material._calculate_n.call_count == 3