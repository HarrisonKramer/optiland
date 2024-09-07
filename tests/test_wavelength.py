import unittest
from optiland.wavelength import Wavelength, WavelengthGroup


class TestWavelength(unittest.TestCase):

    def test_wavelength_initialization(self):
        wl = Wavelength(500, unit='nm')
        self.assertEqual(wl.value, 0.5)
        self.assertEqual(wl.unit, 'um')

    def test_wavelength_conversion(self):
        wl_nm = Wavelength(500, unit='nm')
        wl_um = Wavelength(0.5, unit='um')
        wl_mm = Wavelength(0.0005, unit='mm')
        wl_cm = Wavelength(0.00005, unit='cm')
        wl_m = Wavelength(0.0000005, unit='m')

        self.assertEqual(wl_nm.value, 0.5)
        self.assertEqual(wl_um.value, 0.5)
        self.assertEqual(wl_mm.value, 0.5)
        self.assertEqual(wl_cm.value, 0.5)
        self.assertEqual(wl_m.value, 0.5)

    def test_invalid_unit(self):
        with self.assertRaises(ValueError):
            Wavelength(500, unit='invalid_unit')

    def test_unit_setter(self):
        wl = Wavelength(500, unit='nm')
        self.assertEqual(wl.value, 0.5)
        wl.unit = 'mm'
        self.assertEqual(wl.unit, 'um')
        self.assertEqual(wl._unit, 'mm')
        self.assertEqual(wl.value, 500000)


class TestWavelengthGroup(unittest.TestCase):

    def test_add_wavelength(self):
        wg = WavelengthGroup()
        wg.add_wavelength(500, unit='nm')
        self.assertEqual(wg.num_wavelengths, 1)
        self.assertEqual(wg.get_wavelength(0), 0.5)

    def test_primary_wavelength(self):
        wg = WavelengthGroup()
        wg.add_wavelength(500, unit='nm')
        wg.add_wavelength(600, is_primary=True, unit='nm')
        self.assertEqual(wg.primary_wavelength.value, 0.6)

    def test_multiple_wavelengths(self):
        wg = WavelengthGroup()
        wg.add_wavelength(500, unit='nm')
        wg.add_wavelength(600, unit='nm', is_primary=True)
        self.assertEqual(wg.num_wavelengths, 2)
        self.assertEqual(wg.get_wavelength(0), 0.5)
        self.assertEqual(wg.get_wavelength(1), 0.6)

    def test_get_wavelengths(self):
        wg = WavelengthGroup()
        wg.add_wavelength(500, unit='nm')
        wg.add_wavelength(600, unit='nm')
        self.assertEqual(wg.get_wavelengths(), [0.5, 0.6])
