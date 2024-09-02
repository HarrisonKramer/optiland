"""Optiland Materials Module

This module provides classes related to materials and determination of indices
of refraction and absorption coefficients.

Kramer Harrison, 2024
"""
import abc
import os
from io import StringIO
import yaml
import numpy as np
import pandas as pd


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
        pass  # pragma: no cover

    @abc.abstractmethod
    def k(self, wavelength):
        pass  # pragma: no cover

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
        nD = self.n(0.5875618)
        nF = self.n(0.4861327)
        nC = self.n(0.6562725)
        return (nD - 1) / (nF - nC)


class IdealMaterial(BaseMaterial):
    """
    Represents an ideal material with a fixed refractive index and absorption
    coefficient for all wavelengths.

    Attributes:
        index (float): The refractive index of the material.
        absorp (float): The absorption coefficient of the material.
    """

    def __init__(self, n, k=0):
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
        # TODO - add all formulas
        # TODO - separate formulas into separate methods & use dict for
        # function lookup
        L = wavelength
        C = self.coeffs
        for formula in self.types:
            if formula == 'formula 1':
                try:
                    n = 1 + C[0]
                    for k in range(1, len(C), 2):
                        n += C[k] * L**2 / (L**2 - C[k+1]**2)
                except IndexError:
                    raise ValueError('Invalid coefficients for dispersion '
                                     'formula 1.')
                return np.sqrt(n)

            elif formula == 'formula 2':
                try:
                    n = np.sqrt(1 + C[0] + C[1]*L**2/(L**2 - C[2]) +
                                C[3]*L**2/(L**2 - C[4]) +
                                C[5]*L**2/(L**2 - C[6]))
                except IndexError:
                    raise ValueError('Invalid coefficients for dispersion '
                                     'formula 2.')
                return n

            elif formula == 'formula 3':
                try:
                    n = C[0]
                    for k in range(1, len(C), 2):
                        n += C[k]*L**C[k+1]
                    return np.sqrt(n)
                except IndexError:
                    raise ValueError('Invalid coefficients for dispersion '
                                     'formula 3.')

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
    (optionally) the reference, which is generally the manufacturer name or
    the author name.

    Note:
        The material database is stored in the file `catalog_nk.csv` in the
        `database` directory. This contains the names, references, and
        filenames of the materials.

    Args:
        name (str): The name of the material.
        reference (str, optional): The reference for the material. This is
            generally the manufacturer name, or the author name. The reference
            must be in the filename.
        robust_search (bool, optional): If True, the search will be robust and
            return the first match found. If False, the search will raise an
            error if multiple matches are found.

    Attributes:
        name (str): The name of the material.
        reference (str): The reference for the material.
    """

    _df = None
    _filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '../database/catalog_nk.csv')

    def __init__(self, name, reference=None, robust_search=True):
        self.name = name
        self.reference = reference
        self.robust = robust_search
        file = self._retrieve_file()
        super().__init__(file)

    @classmethod
    def _load_dataframe(cls):
        """Load the DataFrame if not yet loaded."""
        if cls._df is None:
            cls._df = pd.read_csv(cls._filename)
        return cls._df

    @staticmethod
    def _levenshtein_distance(s1, s2):
        """
        Calculates the Levenshtein distance between two strings.

        Args:
            s1 (str): The first string.
            s2 (str): The second string.

        Returns:
            int: The Levenshtein distance between the two strings.
        """
        # Initialize matrix of zeros
        rows = len(s1) + 1
        cols = len(s2) + 1
        distance_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

        # Populate matrix with initial values
        for i in range(1, rows):
            distance_matrix[i][0] = i
        for j in range(1, cols):
            distance_matrix[0][j] = j

        # Calculate the distance
        for i in range(1, rows):
            for j in range(1, cols):
                if s1[i-1] == s2[j-1]:
                    cost = 0
                else:
                    cost = 1
                distance_matrix[i][j] = min(distance_matrix[i-1][j] + 1,
                                            distance_matrix[i][j-1] + 1,
                                            distance_matrix[i-1][j-1] + cost)

        return distance_matrix[-1][-1]

    def _find_material_matches(self, df):
        """
        Finds material matches in a DataFrame based on the given name and
        reference.

        Args:
            df (pandas.DataFrame): The DataFrame containing the materials.

        Returns:
            pandas.DataFrame: A DataFrame containing the filtered materials
                that match the given name and reference.
        """
        # Make input name lowercase
        name = self.name.lower()

        # Filter rows where input string is substring of category_name or name
        dfi = df[
            df['category_name'].str.lower().str.contains(name) |
            df['name'].str.lower().str.contains(name)
        ].copy()

        # If reference given, filter rows non-matching rows
        if self.reference:
            reference = self.reference.lower()
            dfi = dfi[
                dfi['category_name'].str.lower().str.contains(reference) |
                dfi['category_name_full'].str.lower().str.contains(reference) |
                dfi['reference'].str.lower().str.contains(reference) |
                dfi['name'].str.lower().str.contains(reference) |
                dfi['filename'].str.lower().str.contains(reference)
            ]

        # If no rows match, return an empty DataFrame
        if dfi.empty:
            return pd.DataFrame()

        # Calculate similarity scores using Levenshtein distance
        dfi['similarity_score'] = dfi.apply(
            lambda row: min(
                self._levenshtein_distance(name, row['category_name'].lower()),
                self._levenshtein_distance(name, row['name'].lower())
            ), axis=1
        )

        # Sort by similarity score in ascending order
        dfi = dfi.sort_values(by='similarity_score').reset_index(drop=True)

        # Warning if no exact matches found
        if dfi['similarity_score'].iloc[0] > 0:
            print(f'Warning: No exact matches found for material {self.name}. '
                  'Material is likely invalid.')

        return dfi

    def _retrieve_file(self):
        """
        Retrieves the file path for the material based on the given criteria.

        Returns:
            str: The file path for the material.

        Raises:
            ValueError: If no matches are found for the material.
            ValueError: If multiple matches are found for the material.
        """
        df = self._load_dataframe()
        filtered_df = self._find_material_matches(df)

        if filtered_df.empty:
            if self.reference:
                raise ValueError(f'No matches found for material {self.name} '
                                 f'with reference {self.reference}')
            else:
                raise ValueError(f'No matches found for material {self.name}')

        if self.robust:
            filename = filtered_df.loc[0, 'filename']
        else:
            if self.reference:
                raise ValueError(f'Multiple matches found for material '
                                 f'{self.name} with reference '
                                 f'{self.reference}')
            else:
                raise ValueError(f'Multiple matches found for material '
                                 f'{self.name}')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        database_dir = os.path.join(current_dir, '../database/data-nk')
        return os.path.join(database_dir, filename)
