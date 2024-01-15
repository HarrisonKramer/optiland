import numpy as np


class ZemaxFileReader:

    def __init__(self, filename):
        self.filename = filename

        self.data = {}
        self.data['aperture'] = {}
        self.data['fields'] = {}
        self.data['wavelengths'] = {}
        self.data['wavelengths']['data'] = []
        self.data['surfaces'] = {}

        self._current_surf_data = {}

        self._operand_table = {
            'FNUM': self._read_fno,
            'ENVD': self._read_epd,
            'XFLN': self._read_x_fields,
            'YFLN': self._read_y_fields,
            'WAVM': self._read_wavelength,
            'SURF': self._read_surface,
            'CURV': self._read_radius,
            'DISZ': self._read_thickness,
            'GLAS': self._read_glass,
            'STOP': self._read_stop,
        }
        self._current_surf = -1
        self._read_file()

    def generate_lens(self):
        # TODO: convert to optiland optic instance
        pass

    def _read_file(self):
        with open(self.filename, 'r', encoding='utf-16') as file:
            for line in file:
                data = line.split()
                try:
                    operand = data[0]
                    self._operand_table[operand](data)
                except (IndexError, KeyError):
                    continue

        # sort and filter fields
        unique_fields = set()
        for i in range(min(len(self.data['fields']['x']),
                           len(self.data['fields']['y']))):
            pair = (self.data['fields']['x'][i], self.data['fields']['y'][i])
            unique_fields.add(pair)

        # Sort the unique pairs based on the second element
        sorted_fields = sorted(unique_fields, key=lambda x: x[1])

        # Unzip the sorted pairs back into two lists
        self.data['fields']['x'], \
            self.data['fields']['y'] = zip(*sorted_fields)

        # set primary wavelength

    def _read_fno(self, data):
        self.data['aperture']['FNO'] = float(data[1])

    def _read_epd(self, data):
        self.data['aperture']['EPD'] = float(data[1])

    def _read_x_fields(self, data):
        self.data['fields']['x'] = [float(value) for value in data[1:]]

    def _read_y_fields(self, data):
        self.data['fields']['y'] = [float(value) for value in data[1:]]

    def _read_wavelength(self, data):
        # TODO: this is a workaround for now
        value = float(data[2])
        if value != 0.55:
            self.data['wavelengths']['data'].append(value)

    def _read_surface(self, data):
        if self._current_surf >= 0:
            self.data['surfaces'][self._current_surf] = self._current_surf_data

        self._current_surf_data = {}
        self._current_surf_data['is_stop'] = False
        self._current_surf += 1

    def _read_radius(self, data):
        try:
            self._current_surf_data['radius'] = 1 / float(data[1])
        except ZeroDivisionError:
            self._current_surf_data['radius'] = np.inf

    def _read_thickness(self, data):
        if data[1] == 'INFINITY':
            self._current_surf_data['thickness'] = np.inf
        else:
            self._current_surf_data['thickness'] = float(data[1])

    def _read_glass(self, data):
        self._current_surf_data['name'] = data[1]
        self._current_surf_data['index'] = float(data[4])
        self._current_surf_data['abbe'] = float(data[5])

    def _read_stop(self, data):
        self._current_surf_data['is_stop'] = True

    def _read_primary_wave(self, data):
        self.data['wavelengths']['primary_index'] = int(data[1])
