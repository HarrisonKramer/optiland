"""Optiland File Handler Module

This module provides classes for parsing files, such as Zemax lens data files.

Kramer Harrison, 2024
"""
import numpy as np


class ZemaxFileReader:
    """
    A class for reading Zemax files and extracting optical data.

    Args:
        filename (str): The path to the Zemax file to be read.

    Attributes:
        filename (str): The path to the Zemax file.
        data (dict): A dictionary to store the extracted optical data.
            - 'aperture' (dict): Dictionary to store aperture data.
            - 'fields' (dict): Dictionary to store field data.
            - 'wavelengths' (dict): Dictionary to store wavelength data.
                - 'data' (list): List to store wavelength values.
            - 'surfaces' (dict): Dictionary to store surface data.
        _current_surf_data (dict): Temporary storage for current surface data.
        _operand_table (dict): Dictionary mapping operand codes to
            corresponding methods.
        _current_surf (int): Index of the current surface being read.

    Methods:
        generate_lens(): Converts the extracted data into an Optiland optic
            instance.
        _read_file(): Reads the Zemax file and extracts the optical data.
        _read_fno(data): Extracts the FNO (F-number) data.
        _read_epd(data): Extracts the EPD (entrance pupil diameter) data.
        _read_x_fields(data): Extracts the x-field data.
        _read_y_fields(data): Extracts the y-field data.
        _read_wavelength(data): Extracts the wavelength data.
        _read_surface(data): Extracts the surface data.
        _read_radius(data): Extracts the radius data.
        _read_thickness(data): Extracts the thickness data.
        _read_glass(data): Extracts the glass data.
        _read_stop(data): Extracts the stop data.
        _read_primary_wave(data): Extracts the primary wavelength data.
    """

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
        """
        Converts the extracted optical data into an Optiland optic instance.
        """
        # TODO: convert to optiland optic instance
        pass

    def _read_file(self):
        """
        Reads the Zemax file and extracts the optical data.
        """
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
        """
        Extracts the FNO (F-number) data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        self.data['aperture']['FNO'] = float(data[1])

    def _read_epd(self, data):
        """
        Extracts the EPD (entrance pupil diameter) data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        self.data['aperture']['EPD'] = float(data[1])

    def _read_x_fields(self, data):
        """
        Extracts the x-field data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        self.data['fields']['x'] = [float(value) for value in data[1:]]

    def _read_y_fields(self, data):
        """
        Extracts the y-field data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        self.data['fields']['y'] = [float(value) for value in data[1:]]

    def _read_wavelength(self, data):
        """
        Extracts the wavelength data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        # TODO: this is a workaround for now
        value = float(data[2])
        if value != 0.55:
            self.data['wavelengths']['data'].append(value)

    def _read_surface(self, data):
        """
        Extracts the surface data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        if self._current_surf >= 0:
            self.data['surfaces'][self._current_surf] = self._current_surf_data

        self._current_surf_data = {}
        self._current_surf_data['is_stop'] = False
        self._current_surf += 1

    def _read_radius(self, data):
        """
        Extracts the radius data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        try:
            self._current_surf_data['radius'] = 1 / float(data[1])
        except ZeroDivisionError:
            self._current_surf_data['radius'] = np.inf

    def _read_thickness(self, data):
        """
        Extracts the thickness data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        if data[1] == 'INFINITY':
            self._current_surf_data['thickness'] = np.inf
        else:
            self._current_surf_data['thickness'] = float(data[1])

    def _read_glass(self, data):
        """
        Extracts the glass data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        self._current_surf_data['name'] = data[1]
        self._current_surf_data['index'] = float(data[4])
        self._current_surf_data['abbe'] = float(data[5])

    def _read_stop(self, data):
        """
        Extracts the stop data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        self._current_surf_data['is_stop'] = True

    def _read_primary_wave(self, data):
        """
        Extracts the primary wavelength data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        self.data['wavelengths']['primary_index'] = int(data[1])
