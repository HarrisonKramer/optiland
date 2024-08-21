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
            'ENPD': self._read_epd,
            'OBNA': self._read_object_na,
            'FLOA': self._read_floating_stop,
            'FTYP': self._read_field_type,
            'XFLN': self._read_x_fields,
            'YFLN': self._read_y_fields,
            'WAVM': self._read_wavelength,
            'PWAV': self._read_primary_wave,
            'SURF': self._read_surface,
            'TYPE': self._read_surf_type,
            'PARM': self._read_surface_parameter,
            'CURV': self._read_radius,
            'DISZ': self._read_thickness,
            'GLAS': self._read_glass,
            'STOP': self._read_stop,
            'MODE': self._read_mode,
            'GCAT': self._read_glass_catalog,
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
        if int(data[2]) == 0:
            self.data['aperture']['imageFNO'] = float(data[1])
        elif int(data[2]) == 1:
            self.data['aperture']['paraxialImageFNO'] = float(data[1])

    def _read_epd(self, data):
        """
        Extracts the EPD (entrance pupil diameter) data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        self.data['aperture']['EPD'] = float(data[1])

    def _read_object_na(self, data):
        """
        Extracts the object-space numerical aperture.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        if int(data[2]) == 0:
            self.data['aperture']['objectNA'] = float(data[1])
        elif int(data[2]) == 1:
            self.data['aperture']['object_cone_angle'] = float(data[1])

    def _read_floating_stop(self, data):
        """
        Extracts the floating stop data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        self.data['aperture']['floating_stop'] = True

    def _read_field_type(self, data):
        """
        Extracts the field type data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        if int(data[1]) == 0:
            self.data['fields']['type'] = 'angle'
        elif int(data[1]) == 1:
            self.data['fields']['type'] = 'object_height'
        elif int(data[1]) == 2:
            self.data['fields']['type'] = 'paraxial_image_height'
        elif int(data[1]) == 3:
            self.data['fields']['type'] = 'real_image_height'
        elif int(data[1]) == 4:
            self.data['fields']['type'] = 'theodolite_angle'
        else:
            self.data['fields']['type'] = 'unsupported'

        if int(data[2]) == 1:
            self.data['fields']['object_space_telecentric'] = True
        else:
            self.data['fields']['object_space_telecentric'] = False

        if int(data[7]) == 1:
            self.data['fields']['afocal_image_space'] = True
        else:
            self.data['fields']['afocal_image_space'] = False

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
        self._current_surf_data['material'] = data[1]
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
        self.data['wavelengths']['primary_index'] = int(data[1]) - 1

    def _read_mode(self, data):
        """
        Extracts the mode data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        if data[1] != 'SEQ':
            raise ValueError('Only sequential mode is supported.')

    def _read_glass_catalog(self, data):
        """
        Extracts the glass catalog data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        self.data['glass_catalogs'] = data[1:]

    def _read_surf_type(self, data):
        """
        Extracts the surface type data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        type_map = {'STANDARD': 'standard',
                    'EVENASPH': 'even_asphere'}
        try:
            self._current_surf_data['type'] = type_map[data[1]]
        except KeyError:
            self._current_surf_data['type'] = 'unsupported'

    def _read_surface_parameter(self, data):
        """
        Extracts the surface parameter data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        key = f'param_{int(data[1])-1}'
        self._current_surf_data[key] = float(data[2])
