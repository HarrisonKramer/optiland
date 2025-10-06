# tests/materials/test_abbe.py
"""
Tests for the AbbeMaterial class in optiland.materials.
"""
import pytest

from optiland import materials
from ..utils import assert_allclose


@pytest.fixture
def abbe_material():
    """
    Provides an instance of AbbeMaterial for testing.
    """
    return materials.AbbeMaterial(n=1.5, abbe=50)


def test_refractive_index(set_test_backend, abbe_material):
    """
    Tests the calculation of the refractive index at a specific wavelength.
    """
    wavelength = 0.58756  # in microns
    value = abbe_material.n(wavelength)
    assert_allclose(value, 1.4999167964912952)


def test_extinction_coefficient(set_test_backend, abbe_material):
    """
    Tests that the extinction coefficient is always zero for an AbbeMaterial.
    """
    wavelength = 0.58756  # in microns
    assert abbe_material.k(wavelength) == 0


def test_coefficients(set_test_backend, abbe_material):
    """
    Tests that the correct number of coefficients are generated for the
    internal polynomial calculation.
    """
    coefficients = abbe_material._get_coefficients()
    assert coefficients.shape == (4,)  # Assuming the polynomial is of degree 3


def test_abbe_to_dict(set_test_backend, abbe_material):
    """
    Tests the serialization of an AbbeMaterial instance to a dictionary.
    """
    abbe_dict = abbe_material.to_dict()
    assert abbe_dict == {"type": "AbbeMaterial", "index": 1.5, "abbe": 50}


def test_abbe_from_dict(set_test_backend):
    """
    Tests the deserialization of an AbbeMaterial instance from a dictionary.
    """
    abbe_dict = {"type": "AbbeMaterial", "index": 1.5, "abbe": 50}
    abbe_material = materials.BaseMaterial.from_dict(abbe_dict)
    assert abbe_material.index == 1.5
    assert abbe_material.abbe == 50


def test_abbe_out_of_bounds_wavelength(set_test_backend):
    """
    Tests that requesting the refractive index outside of the valid
    wavelength range raises a ValueError.
    """
    abbe_material = materials.AbbeMaterial(n=1.5, abbe=50)
    with pytest.raises(ValueError):
        abbe_material.n(0.3)
    with pytest.raises(ValueError):
        abbe_material.n(0.8)