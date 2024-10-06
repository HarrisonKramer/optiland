import pytest
import numpy as np
from optiland.materials.abbe import AbbeMaterial


@pytest.fixture
def abbe_material():
    return AbbeMaterial(n=1.5, abbe=50)


def test_refractive_index(abbe_material):
    wavelength = 0.58756  # in microns
    value = abbe_material.n(wavelength)
    assert pytest.approx(value, rel=1e-5) == 1.4999167964912952


def test_absorption_coefficient(abbe_material):
    wavelength = 0.58756  # in microns
    assert abbe_material.k(wavelength) == 0


def test_coefficients(abbe_material):
    coefficients = abbe_material._get_coefficients()
    assert isinstance(coefficients, np.ndarray)
    assert coefficients.shape == (4,)  # Assuming the polynomial is of degree 3


def test_refractive_index_different_wavelengths(abbe_material):
    wavelengths = [0.4, 0.5, 0.6, 0.7]  # in microns
    for wavelength in wavelengths:
        assert isinstance(abbe_material.n(wavelength), float)
