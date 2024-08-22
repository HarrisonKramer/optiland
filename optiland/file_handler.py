"""Optiland File Handler Module

This module provides classes for parsing files, such as Zemax lens data files.

Kramer Harrison, 2024
"""
import os
import re
import requests
import tempfile
import numpy as np
from optiland.optic import Optic
from optiland.materials import Material


class ZemaxFileReader:
    """
    A class for reading Zemax files and extracting optical data.

    Args:
        source (str): The source of the .zmx file, either a filename or a URL.

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
    """

    def __init__(self, source):
        self.source = source

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
            'FTYP': self._read_config_data,
            'XFLN': self._read_x_fields,
            'YFLN': self._read_y_fields,
            'WAVM': self._read_wavelength,
            'PWAV': self._read_primary_wave,
            'SURF': self._read_surface,
            'TYPE': self._read_surf_type,
            'PARM': self._read_surface_parameter,
            'CURV': self._read_radius,
            'DISZ': self._read_thickness,
            'CONI': self._read_conic,
            'GLAS': self._read_glass,
            'STOP': self._read_stop,
            'MODE': self._read_mode,
            'GCAT': self._read_glass_catalog,
        }

        self._current_surf = -1
        self._configure_source_input()
        self._read_file()

    def generate_lens(self):
        """
        Converts the extracted optical data into an Optiland optic instance.
        """
        converter = ZemaxToOpticConverter(self.data)
        return converter.convert()

    def _is_url(self, source):
        """
        Check if the source is a URL.

        Args:
            source (str): The source to check.

        Returns:
            bool: True if the source is a URL, False otherwise.
        """
        return re.match(r'^https?://', source) is not None

    def _configure_source_input(self):
        """
        Checks if the source is a URL and writes to a temporary file if so.
        Otherwise, sets the source to the filename.
        """
        if self._is_url(self.source):
            response = requests.get(self.source)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False) as file:
                    file.write(response.content)
                    self.filename = file.name
            else:
                raise ValueError('Failed to download Zemax file.')
        else:
            self.filename = self.source

    def _read_file(self):
        """
        Reads the Zemax file and extracts the optical data.
        """
        encodings = ['utf-16', 'utf-8']
        success = False
        for encoding in encodings:
            try:
                with open(self.filename, 'r', encoding=encoding) as file:
                    for line in file:
                        data = line.split()
                        try:
                            operand = data[0]
                            self._operand_table[operand](data)
                        except (IndexError, KeyError):
                            continue
            except (UnicodeError, UnicodeDecodeError):
                continue

            if self.data['aperture']:
                success = True
            else:
                continue

        if not success:
            raise ValueError('Failed to read Zemax file.')

        # sort and filter fields
        unique_fields = set()
        for i in range(min(len(self.data['fields']['x']),
                           len(self.data['fields']['y']))):
            pair = (self.data['fields']['x'][i], self.data['fields']['y'][i])
            unique_fields.add(pair)

        # Sort the unique field pairs based on the second element
        sorted_fields = sorted(unique_fields, key=lambda x: x[1])

        # Unzip the sorted pairs back into two lists for x, y fields
        self.data['fields']['x'], \
            self.data['fields']['y'] = zip(*sorted_fields)

        # remove temporary file if it was created
        if self._is_url(self.source):
            os.remove(self.filename)

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

    def _read_config_data(self, data):
        """
        Extracts field, wavelength, and other configuration data.

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

        self.data['wavelengths']['num_wavelengths'] = int(data[4])

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
        value = float(data[2])
        num_wavelengths = self.data['wavelengths']['num_wavelengths']
        if len(self.data['wavelengths']['data']) < num_wavelengths:
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
        self._current_surf_data['type'] = 'standard'
        self._current_surf_data['is_stop'] = False
        self._current_surf_data['conic'] = 0.0
        self._current_surf_data['material'] = 'air'
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

    def _read_conic(self, data):
        """
        Extracts the conic constant data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        self._current_surf_data['conic'] = float(data[1])

    def _read_glass(self, data):
        """
        Extracts the glass data.

        Args:
            data (list): List of data values extracted from the Zemax file.
        """
        material = data[1]
        self._current_surf_data['material'] = material

        # Generate a Material object from the material name & manufacturer
        try:
            # Try to create a Material object from the material name
            self._current_surf_data['material'] = Material(material)
        except ValueError:

            # If the material name is not recognized, try to create a Material
            # object from the material name and manufacturer
            for manufacturer in self.data['glass_catalogs']:
                try:
                    self._current_surf_data['material'] = \
                        Material(material, manufacturer.lower())
                    break
                except ValueError:
                    continue

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


class ZemaxToOpticConverter:
    """Converts Zemax data into an Optic object.

    Args:
        zemax_data (dict): The Zemax data to be converted. This is the data
            extracted by the ZemaxFileReader.

    Attributes:
        data (dict): The Zemax data to be converted.
        optic (Optic): The Optic object based on the Zemax data.

    Methods:
        convert(): Converts the configuration of the file handler into an
            Optic object.
    """

    def __init__(self, zemax_data):
        self.data = zemax_data
        self.optic = None

    def convert(self):
        """
        Converts the configuration of the file handler into an Optic object.

        Returns:
            Optic: The Optic object based on the Zemax data.
        """
        self.optic = Optic()
        self._configure_surfaces()
        self._configure_aperture()
        self._configure_fields()
        self._configure_wavelengths()
        return self.optic

    def _configure_surfaces(self):
        """
        Configures the surfaces for the optic.
        """
        for idx, surf_data in self.data['surfaces'].items():
            self._configure_surface(idx, surf_data)
        self.optic.add_surface(index=len(self.data['surfaces']))

    def _configure_surface(self, index, data):
        """
        Configures a surface for the optic.
        """
        coefficients = self._configure_surface_coefficients(data)
        self.optic.add_surface(index=index,
                               surface_type=data['type'],
                               radius=data['radius'],
                               conic=data['conic'],
                               thickness=data['thickness'],
                               is_stop=data['is_stop'],
                               material=data['material'],
                               coefficients=coefficients)

    def _configure_surface_coefficients(self, data):
        """
        Configures the coefficients for the surface. This is None for standard
        surfaces.
        """
        surf_type = data['type']
        if surf_type == 'standard':
            return None
        elif surf_type == 'even_asphere':
            coefficients = []
            for k in range(8):
                coefficients.append(data[f'param_{k}'])
            return coefficients
        else:
            raise ValueError('Unsupported surface type.')

    def _configure_aperture(self):
        """
        Configures the aperture for the optic.
        """
        aperture_data = self.data['aperture']
        ap_type, value = next(iter(aperture_data.items()))
        self.optic.set_aperture(aperture_type=ap_type, value=value)

    def _configure_fields(self):
        """
        Configure the fields for the optic.
        """
        self.optic.set_field_type(field_type=self.data['fields']['type'])
        for fx, fy in zip(self.data['fields']['x'], self.data['fields']['y']):
            self.optic.add_field(x=fx, y=fy)

    def _configure_wavelengths(self):
        """
        Configure the wavelengths for the optic.
        """
        primary_idx = self.data['wavelengths']['primary_index']
        for idx, value in enumerate(self.data['wavelengths']['data']):
            self.optic.add_wavelength(value=value,
                                      is_primary=(idx == primary_idx))
