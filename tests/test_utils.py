import pytest
from optiland.optic import Optic
from optiland.utils import resolve_fields, resolve_wavelength, resolve_wavelengths


@pytest.fixture
def optic():
    o = Optic()
    o.add_wavelength(value=0.5, is_primary=True)
    o.add_wavelength(value=0.6)
    o.add_field(y=0)
    o.add_field(y=1)
    return o


def test_resolve_wavelengths_all(optic):
    assert resolve_wavelengths(optic, "all") == [0.5, 0.6]


def test_resolve_wavelengths_primary(optic):
    assert resolve_wavelengths(optic, "primary") == [0.5]


def test_resolve_wavelengths_list(optic):
    assert resolve_wavelengths(optic, [0.7, 0.8]) == [0.7, 0.8]


def test_resolve_wavelengths_invalid_string(optic):
    with pytest.raises(ValueError):
        resolve_wavelengths(optic, "invalid")


def test_resolve_wavelengths_invalid_type(optic):
    with pytest.raises(TypeError):
        resolve_wavelengths(optic, 123)


def test_resolve_fields_all(optic):
    assert resolve_fields(optic, "all") == [(0.0, 0.0), (0.0, 1.0)]


def test_resolve_fields_list(optic):
    assert resolve_fields(optic, [(0.1, 0.2)]) == [(0.1, 0.2)]


def test_resolve_fields_invalid_string(optic):
    with pytest.raises(ValueError):
        resolve_fields(optic, "primary")


def test_resolve_fields_invalid_type(optic):
    with pytest.raises(TypeError):
        resolve_fields(optic, 123)


def test_resolve_wavelength_primary(optic):
    assert resolve_wavelength(optic, "primary") == 0.5


def test_resolve_wavelength_float(optic):
    assert resolve_wavelength(optic, 0.7) == 0.7


def test_resolve_wavelength_int(optic):
    assert resolve_wavelength(optic, 1) == 1.0


def test_resolve_wavelength_invalid_string(optic):
    with pytest.raises(ValueError):
        resolve_wavelength(optic, "all")


def test_resolve_wavelength_invalid_type(optic):
    with pytest.raises(TypeError):
        resolve_wavelength(optic, [0.5])