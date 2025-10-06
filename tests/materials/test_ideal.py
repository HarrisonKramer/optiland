# tests/materials/test_ideal.py
"""
Tests for the IdealMaterial class in optiland.materials.
"""
import optiland.backend as be
import pytest

from optiland import materials


class TestIdealMaterial:
    """
    Tests for the IdealMaterial class, which represents a material with a
    constant refractive index and extinction coefficient.
    """
    def test_ideal_material_n(self, set_test_backend):
        """
        Verifies that the refractive index 'n' is constant across different
        wavelengths.
        """
        material = materials.IdealMaterial(n=1.5)
        assert material.n(0.5) == 1.5
        assert material.n(1.0) == 1.5
        assert material.n(2.0) == 1.5
        assert material.k(2.0) == 0.0

    def test_ideal_material_k(self, set_test_backend):
        """
        Verifies that the extinction coefficient 'k' is constant across
        different wavelengths.
        """
        material = materials.IdealMaterial(n=1.5, k=0.2)
        assert material.k(0.5) == 0.2
        assert material.k(1.0) == 0.2
        assert material.k(2.0) == 0.2

    def test_ideal_to_dict(self, set_test_backend):
        """
        Tests the serialization of an IdealMaterial instance to a dictionary.
        """
        material = materials.IdealMaterial(n=1.5, k=0.2)
        assert material.to_dict() == {
            "index": 1.5,
            "absorp": 0.2,
            "type": materials.IdealMaterial.__name__,
        }

    def test_ideal_from_dict(self, set_test_backend):
        """
        Tests the deserialization of an IdealMaterial instance from a
        dictionary.
        """
        material = materials.IdealMaterial.from_dict(
            {"index": 1.5, "absorp": 0.2, "type": materials.IdealMaterial.__name__},
        )
        assert material.n(0.5) == 1.5
        assert material.k(0.5) == 0.2