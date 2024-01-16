class Wavelength:

    def __init__(self, value, is_primary=True, unit='microns'):
        self._value = value
        self.is_primary = is_primary
        self._unit = unit.lower()
        self._value_in_um = self._convert_to_um()

    @property
    def value(self):
        return self._value_in_um

    @property
    def unit(self):
        return 'um'

    @unit.setter
    def unit(self, new_unit):
        self._unit = new_unit.lower()

    def _convert_to_um(self):
        unit_conversion = {
            'nm': 0.001,
            'um': 1,
            'mm': 1000,
            'cm': 10000,
            'm': 1000000
        }

        if self._unit in unit_conversion:
            conversion_factor = unit_conversion[self._unit]
            return self._value * conversion_factor
        else:
            raise ValueError('Unsupported unit for conversion to microns.')


class WavelengthGroup:

    def __init__(self):
        self.wavelengths = []

    @property
    def num_wavelengths(self):
        return len(self.wavelengths)

    @property
    def primary_index(self):
        for index, wavelength in enumerate(self.wavelengths):
            if wavelength.is_primary:
                return index

    @property
    def primary_wavelength(self):
        return self.wavelengths[self.primary_index]

    def add_wavelength(self, value, is_primary=True, unit='um'):
        if is_primary:
            for wavelength in self.wavelengths:
                wavelength.is_primary = False

        if self.num_wavelengths == 0:
            is_primary = True

        self.wavelengths.append(Wavelength(value, is_primary, unit))

    def get_wavelength(self, wavelength_number):
        return self.wavelengths[wavelength_number].value

    def get_wavelengths(self):
        return [wave.value for wave in self.wavelengths]
