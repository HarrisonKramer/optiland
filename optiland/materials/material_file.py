import os
from io import StringIO
import yaml
import numpy as np
from optiland.materials.base import BaseMaterial


class MaterialFile(BaseMaterial):
    """
    Represents a material based on a material YAML file from the
    refractiveindex.info database.

    Material refractive indices are based on various dispersion formulas or
    tabulated data. The material file contains the coefficients for the
    dispersion formulas and/or tabulated data.

    See https://refractiveindex.info/database/doc/Dispersion%20formulas.pdf

    Args:
        filename (str): The path to the material file.

    Attributes:
        filename (str): The filename of the material file.
        coefficients (list): A list of coefficients for calculating the
            refractive index.

    Methods:
        n(wavelength): Calculates the refractive index of the material at a
            given wavelength.
        k(wavelength): Retrieves the extinction coefficient of the material at
            a given wavelength.
    """
    def __init__(self, filename):
        self.filename = filename

        self.coefficients = []
        self._k_wavelength = None
        self._k = None
        self._n_formula = None
        self._n_wavelength = None
        self._n = None
        self.reference_data = None

        self.formula_map = {'formula 1': self._formula_1,
                            'formula 2': self._formula_2,
                            'formula 3': self._formula_3,
                            'formula 4': self._formula_4,
                            'formula 5': self._formula_5,
                            'formula 6': self._formula_6,
                            'formula 7': self._formula_7,
                            'formula 8': self._formula_8,
                            'formula 9': self._formula_9,
                            'tabulated n': self._tabulated_n,
                            'tabulated nk': self._tabulated_n}

        data = self._read_file()
        self._parse_file(data)

    def n(self, wavelength):
        """
        Calculates the refractive index of the material at given wavelengths.

        Args:
            wavelength (float or numpy.ndarray): The wavelength(s) in microns.

        Returns:
            float or numpy.ndarray: The refractive index(s) of the material.
        """
        func = self.formula_map[self._n_formula]
        return func(wavelength)

    def k(self, wavelength):
        """
        Retrieves the extinction coefficient of the material at a
        given wavelength.

        Args:
            wavelength (float or numpy.ndarray): The wavelength(s) in microns.

        Returns:
            float or numpy.ndarray: The extinction coefficient of the material.

        Raises:
            ValueError: If no extinction coefficient data is found.
        """
        try:
            return np.interp(wavelength, self._k_wavelength, self._k)
        except ValueError:
            file = os.path.basename(self.filename)
            raise ValueError(f'No extinction coefficient data found for '
                             f'{file}.')

    def _formula_1(self, w):
        """
        Calculate the refractive index using dispersion formula 1 from
        refractiveindex.info (Sellmeier formula).

        Args:
            wavelength (float or numpy.ndarray): The wavelength(s) in microns.

        Returns:
            float or numpy.ndarray: The refractive index(s) of the material.
        """
        c = self.coefficients
        try:
            n = 1 + c[0]
            for k in range(1, len(c), 2):
                n += c[k] * w**2 / (w**2 - c[k+1]**2)
        except IndexError:
            raise ValueError('Invalid coefficients for dispersion formula 1.')
        return np.sqrt(n)

    def _formula_2(self, w):
        """
        Calculate the refractive index using dispersion formula 2 from
        refractiveindex.info (Sellmeier-2 formula).

        Args:
            wavelength (float or numpy.ndarray): The wavelength(s) in microns.

        Returns:
            float or numpy.ndarray: The refractive index(s) of the material.
        """
        c = self.coefficients
        try:
            n = 1 + c[0]
            for k in range(1, len(c), 2):
                n += c[k] * w**2 / (w**2 - c[k+1])
        except IndexError:
            raise ValueError('Invalid coefficients for dispersion formula 2.')
        return np.sqrt(n)

    def _formula_3(self, w):
        """
        Calculate the refractive index using dispersion formula 3 from
        refractiveindex.info (Polynomial formula).

        Args:
            wavelength (float or numpy.ndarray): The wavelength(s) in microns.

        Returns:
            float or numpy.ndarray: The refractive index(s) of the material.
        """
        c = self.coefficients
        try:
            n = c[0]
            for k in range(1, len(c), 2):
                n += c[k]*w**c[k+1]
            return np.sqrt(n)
        except IndexError:
            raise ValueError('Invalid coefficients for dispersion formula 3.')

    def _formula_4(self, w):
        """
        Calculate the refractive index using dispersion formula 4 from
        refractiveindex.info (RefractiveIndex.INFO formula).

        Args:
            wavelength (float or numpy.ndarray): The wavelength(s) in microns.

        Returns:
            float or numpy.ndarray: The refractive index(s) of the material.
        """
        c = self.coefficients
        try:
            n = c[0] + c[1]*w**c[2] / (w**2 - c[3]**c[4]) + \
                c[5]*w**c[6] / (w**2 - c[7]**c[8])
            for k in range(9, len(c), 2):
                n += c[k]*w**c[k+1]
            return np.sqrt(n)
        except IndexError:
            raise ValueError('Invalid coefficients for dispersion formula 4.')

    def _formula_5(self, w):
        """
        Calculate the refractive index using dispersion formula 5 from
        refractiveindex.info (Cauchy formula).

        Args:
            wavelength (float or numpy.ndarray): The wavelength(s) in microns.

        Returns:
            float or numpy.ndarray: The refractive index(s) of the material.
        """
        c = self.coefficients
        try:
            n = c[0]
            for k in range(1, len(c), 2):
                n += c[k]*w**c[k+1]
            return n
        except IndexError:
            raise ValueError('Invalid coefficients for dispersion formula 5.')

    def _formula_6(self, w):
        """
        Calculate the refractive index using dispersion formula 6 from
        refractiveindex.info (Gases formula).

        Args:
            wavelength (float or numpy.ndarray): The wavelength(s) in microns.

        Returns:
            float or numpy.ndarray: The refractive index(s) of the material.
        """
        c = self.coefficients
        try:
            n = 1 + c[0]
            for k in range(1, len(c), 2):
                n += c[k] / (c[k+1] - w**-2)
            return n
        except IndexError:
            raise ValueError('Invalid coefficients for dispersion formula 6.')

    def _formula_7(self, w):
        """
        Calculate the refractive index using dispersion formula 7 from
        refractiveindex.info (Herzberger formula).

        Args:
            wavelength (float or numpy.ndarray): The wavelength(s) in microns.

        Returns:
            float or numpy.ndarray: The refractive index(s) of the material.
        """
        c = self.coefficients
        try:
            n = c[0] + c[1] / (w**2 - 0.028) + c[2] * (1 / (w**2 - 0.028))**2
            for k in range(3, len(c)):
                n += c[k] * w**(2*(k-2))
            return n
        except IndexError:
            raise ValueError('Invalid coefficients for dispersion formula 7.')

    def _formula_8(self, w):
        """
        Calculate the refractive index using dispersion formula 8 from
        refractiveindex.info (Retro formula).

        Args:
            wavelength (float or numpy.ndarray): The wavelength(s) in microns.

        Returns:
            float or numpy.ndarray: The refractive index(s) of the material.
        """
        c = self.coefficients
        if len(c) != 4:
            raise ValueError('Invalid coefficients for dispersion formula 8.')

        b = c[0] + c[1] * w**2 / (w**2 - c[2]) + c[3] * w**2
        return np.sqrt((1 + 2*b) / (1 - b))

    def _formula_9(self, w):
        """
        Calculate the refractive index using dispersion formula 9 from
        refractiveindex.info (Exotic formula).

        Args:
            wavelength (float or numpy.ndarray): The wavelength(s) in microns.

        Returns:
            float or numpy.ndarray: The refractive index(s) of the material.
        """
        c = self.coefficients
        if len(c) != 6:
            raise ValueError('Invalid coefficients for dispersion formula 9.')

        n = (c[0] + c[1] / (w**2 - c[2]) +
             c[3]*(w - c[4]) / ((w - c[4])**2 + c[5]))
        return np.sqrt(n)

    def _tabulated_n(self, w):
        """Calculate the refractive index using tabulated data."""
        try:
            return np.interp(w, self._n_wavelength, self._n)
        except ValueError:
            raise ValueError('No tabular refractive index data found.')

    def _read_file(self):
        """Read the material file."""
        with open(self.filename, 'r') as stream:
            return yaml.safe_load(stream)

    def _set_formula_type(self, formula_type):
        """Set the refractive index formula type."""
        if self._n_formula is None:
            self._n_formula = formula_type
        else:
            raise ValueError('Multiple refractive index formulas found.')

    def _parse_file(self, data):
        """Parse the material file data."""
        for sub_data in data['DATA']:
            sub_data_type = sub_data['type']

            # Parse the data based on the type
            if sub_data_type.startswith('formula '):
                self.coefficients = \
                    [float(k) for k in sub_data['coefficients'].split()]
                self._set_formula_type(sub_data_type)

            # Parse tabulated data
            elif sub_data_type.startswith('tabulated'):
                data_file = StringIO(sub_data['data'])
                arr = np.atleast_2d(np.loadtxt(data_file))

                if sub_data_type == 'tabulated n':
                    self._n_wavelength = arr[:, 0]
                    self._n = arr[:, 1]
                    self._set_formula_type(sub_data_type)

                elif sub_data_type == 'tabulated k':
                    self._k_wavelength = arr[:, 0]
                    self._k = arr[:, 1]

                elif sub_data_type == 'tabulated nk':
                    self._n_wavelength = arr[:, 0]
                    self._k_wavelength = arr[:, 0]
                    self._n = arr[:, 1]
                    self._k = arr[:, 2]
                    self._set_formula_type(sub_data_type)

        # Parse reference info, if available
        try:
            self.reference_data = data['REFERENCE']
        except KeyError:
            pass