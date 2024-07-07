"""Optiland Materials Module

This module provides classes related to materials and determination of indices
of refraction and absorption coefficients.

Kramer Harrison, 2024
"""
import abc
import glob
import os
from io import StringIO
import yaml
import numpy as np


class BaseMaterial:
    """
    Base class for materials.

    This class defines the interface for material properties such as
    refractive index (n) and extinction coefficient (k). It also provides a
    method to calculate the Abbe number.

    Subclasses of BaseMaterial should implement the abstract methods `n` and
    `k` to provide specific material properties.

    Attributes:
        None

    Methods:
        n(wavelength): Abstract method to calculate the refractive index at a
            given wavelength in microns.
        k(wavelength): Abstract method to calculate the extinction coefficient
            at a given wavelength in microns.
        abbe(): Method to calculate the Abbe number of the material.
    """
    @abc.abstractmethod
    def n(self, wavelength):
        return

    @abc.abstractmethod
    def k(self, wavelength):
        return

    def abbe(self):
        """
        Calculate the Abbe number of the material.

        The Abbe number is a measure of the dispersion of a material, defined
        as the difference in refractive index between the D-line
        (wavelength = 0.5893 μm) and the F-line (wavelength = 0.4861 μm)
        divided by the difference between the F-line and the C-line
        (wavelength = 0.6563 μm).

        Returns:
            The Abbe number of the material.
        """
        nD = self.n(0.5893)
        nF = self.n(0.4861)
        nC = self.n(0.6563)
        return (nD - 1) / (nF - nC)


class IdealMaterial(BaseMaterial):
    """
    Represents an ideal material with a fixed refractive index and absorption
    coefficient for all wavelengths.

    Attributes:
        index (float): The refractive index of the material.
        absorp (float): The absorption coefficient of the material.
    """

    def __init__(self, n, k):
        self.index = n
        self.absorp = k

    def n(self, wavelength):
        """
        Returns the refractive index of the material.

        Args:
            wavelength (float): The wavelength of light in microns.

        Returns:
            float: The refractive index of the material.
        """
        return self.index

    def k(self, wavelength):
        """
        Returns the absorption coefficient of the material.

        Args:
            wavelength (float): The wavelength of light in microns.

        Returns:
            float: The absorption coefficient of the material.
        """
        return self.absorp


class Mirror(IdealMaterial):
    """
    Represents a mirror material.

    Inherits from the IdealMaterial class.

    Attributes:
        n (float): The refractive index of the material.
        k (float): The extinction coefficient of the material.
    """

    def __init__(self):
        super().__init__(n=-1.0, k=0.0)


class MaterialFile(BaseMaterial):
    """
    Represents a material file containing optical properties.

    Args:
        filename (str): The path to the material file.

    Attributes:
        file (str): The filename of the material file.
        types (list): A list of index formula types.
        coeffs (list): A list of coefficients for calculating the refractive
            index.
        _k_wavelength (numpy.ndarray): The wavelengths for which extinction
            coefficients are available.
        _k (numpy.ndarray): The extinction coefficients corresponding to the
            wavelengths.

    Methods:
        n(wavelength): Calculates the refractive index of the material at a
            given wavelength.
        k(wavelength): Retrieves the extinction coefficient of the material at
            a given wavelength.

    Protected Methods:
        _read_yaml(): Reads the material file in YAML format.
        _decipher_type(): Extracts the optical property types from the
            material file.
        _get_coeffs(): Extracts the coefficients for calculating the
            refractive index from the material file.
        _get_extinction_coeffs(): Extracts the extinction coefficients from
            the material file.
    """

    def __init__(self, filename):
        self.file = filename

        self.types = []
        self.coeffs = []

        self._k_wavelength = None
        self._k = None

        self._read_yaml()
        self._decipher_type()
        self._get_coeffs()
        self._get_extinction_coeffs()

    def n(self, wavelength):
        """
        Calculates the refractive index of the material at a given wavelength.

        Args:
            wavelength (float): The wavelength in microns.

        Returns:
            float: The refractive index of the material.
        """
        L = wavelength
        C = self.coeffs
        for formula in self.types:
            if formula == 'formula 1':
                n = 1 + C[0]
                for k in range(8):
                    n += C[2*k] * L**2 / (L**2 - C[2*k+1]**2)
                return np.sqrt(n)

            elif formula == 'formula 2':
                n = np.sqrt(1 + C[0] + C[1]*L**2/(L**2 - C[2]) +
                            C[3]*L**2/(L**2 - C[4]) + C[5]*L**2/(L**2 - C[6]))
                return n

            elif formula == 'formula 3':
                n = C[0]
                for k in range(1, len(C), 2):
                    n += C[k]*L**C[k+1]
                return np.sqrt(n)

            else:
                return None

    def k(self, wavelength):
        """
        Retrieves the extinction coefficient of the material at a
        given wavelength.

        Args:
            wavelength (float): The wavelength in microns.

        Returns:
            float: The extinction coefficient of the material.

        Raises:
            ValueError: If no extinction coefficient data is found.
        """
        try:
            return np.interp(wavelength, self._k_wavelength, self._k)
        except ValueError:
            raise ValueError('No extinction coefficient data found.')

    def _read_yaml(self):
        """
        Reads the material file in YAML format.
        """
        with open(self.file, 'r') as stream:
            self.data = yaml.safe_load(stream)

        for each in self.data['DATA']:
            if each['type'] is not None:
                self.types.append(each['type'])

    def _decipher_type(self):
        """
        Extracts the optical property types from the material file.
        """
        for each in self.data['DATA']:
            if each['type'] is not None:
                self.types.append(each['type'])

    def _get_coeffs(self):
        """
        Extracts the coefficients for calculating the refractive index from
        the material file.
        """
        self.coeffs = [float(k) for k in
                       self.data['DATA'][0]['coefficients'].split()]

    def _get_extinction_coeffs(self):
        """
        Extracts the extinction coefficients from the material file.
        """
        for data in self.data['DATA']:
            if data['type'] == 'tabulated k':
                data_file = StringIO(self.data['DATA'][1]['data'])
                k_data = np.loadtxt(data_file)
                self._k_wavelength = k_data[:, 0]
                self._k = k_data[:, 1]


class Material(MaterialFile):
    """
    Represents a generic material used in the Optiland system.
    This class identifies the correct material given the material name and
    (optionally) the reference, or manufacturer.

    Args:
        name (str): The name of the material.
        reference (str, optional): The reference for the material. This is
            generally the manufacturer name, or the author name. The reference
            must be in the filename.

    Attributes:
        name (str): The name of the material.
        reference (str): The reference for the material.
    """

    def __init__(self, name, reference=None):
        self.name = name
        self.reference = reference
        file = self._retrieve_file()
        super().__init__(file)

    def _retrieve_file(self):
        """
        Retrieves the file path for the material.

        Returns:
            str: The file path for the material.

        Raises:
            ValueError: If no glass data is found for the material.
            ValueError: If more than one material manufacturer is found for
                the material.
        """
        search_paths = [os.path.join(os.path.dirname(__file__), '..',
                                     f'database/**/{self.name}/**/*.yml'),
                        os.path.join(os.path.dirname(__file__), '..',
                                     f'database/**/{self.name}.yml')]

        files = []
        for path in search_paths:
            for filename in glob.iglob(path, recursive=True):
                files.append(filename)

        if self.reference:
            files = [file for file in files if self.reference in file]

        if not files:
            raise ValueError(f'No glass data found for "{self.name}"')

        if len(files) > 1:
            error_str = f'''More than one material manufacturer found for
            {self.name}: {files}. Please additionally list manufacturer.
            '''
            raise ValueError(error_str)

        return files[0]
