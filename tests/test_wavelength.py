import pytest
from optiland.wavelength import Wavelength, WavelengthGroup


class TestWavelengths:
    def test_wavelength_initialization(self):
        wl = Wavelength(500, unit='nm')
        assert wl.value == 0.5
        assert wl.unit == 'um'

    def test_wavelength_conversion(self):
        wl_nm = Wavelength(500, unit='nm')
        wl_um = Wavelength(0.5, unit='um')
        wl_mm = Wavelength(0.0005, unit='mm')
        wl_cm = Wavelength(0.00005, unit='cm')
        wl_m = Wavelength(0.0000005, unit='m')

        assert wl_nm.value == 0.5
        assert wl_um.value == 0.5
        assert wl_mm.value == 0.5
        assert wl_cm.value == 0.5
        assert wl_m.value == 0.5

    def test_invalid_unit(self):
        with pytest.raises(ValueError):
            Wavelength(500, unit='invalid_unit')

    def test_unit_setter(self):
        wl = Wavelength(500, unit='nm')
        assert wl.value == 0.5
        wl.unit = 'mm'
        assert wl.unit == 'um'
        assert wl._unit == 'mm'
        assert wl.value == 500000


class TestWavelengthGroups:
    def test_add_wavelength(self):
        wg = WavelengthGroup()
        wg.add_wavelength(500, unit='nm')
        assert wg.num_wavelengths == 1
        assert wg.get_wavelength(0) == 0.5

    def test_primary_wavelength(self):
        wg = WavelengthGroup()
        wg.add_wavelength(500, unit='nm')
        wg.add_wavelength(600, is_primary=True, unit='nm')
        assert wg.primary_wavelength.value == 0.6

    def test_multiple_wavelengths(self):
        wg = WavelengthGroup()
        wg.add_wavelength(500, unit='nm')
        wg.add_wavelength(600, unit='nm', is_primary=True)
        assert wg.num_wavelengths == 2
        assert wg.get_wavelength(0) == 0.5
        assert wg.get_wavelength(1) == 0.6

    def test_get_wavelengths(self):
        wg = WavelengthGroup()
        wg.add_wavelength(500, unit='nm')
        wg.add_wavelength(600, unit='nm')
        assert wg.get_wavelengths() == [0.5, 0.6]