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
            raise ValueError('No extinction coefficient data found.')

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

    def __init__(self, name, reference=None, robust_search=True,
                 min_wavelength=None, max_wavelength=None):
        self.name = name
        self.reference = reference
        self.robust = robust_search
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        file, self.material_data = self._retrieve_file()
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

        # Filter rows based on wavelength range
        if self.min_wavelength:
            dfi = dfi[(dfi['min_wavelength'] <= self.min_wavelength) &
                      (dfi['max_wavelength'] >= self.min_wavelength)]
        if self.max_wavelength:
            dfi = dfi[(dfi['min_wavelength'] <= self.max_wavelength) &
                      (dfi['max_wavelength'] >= self.max_wavelength)]

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
                  'Material may be invalid.')

        return dfi

    def _raise_material_error(self, no_matches=False, multiple_matches=False):
        """
        Raises an error if no matches or multiple matches are found for the
        material.

        Args:
            no_matches (bool): Indicates if no matches were found.
            multiple_matches (bool): Indicates if multiple matches were found.

        Raises:
            ValueError: If no matches or multiple matches are found for the
                material.
        """
        if no_matches:
            message = f'No matches found for material {self.name}'
        elif multiple_matches:
            message = f'Multiple matches found for material {self.name}'
        else:
            message = f'Error finding material {self.name}'

        if self.reference:
            message += f' with reference {self.reference}'

        if self.min_wavelength or self.max_wavelength:
            wavelength_range = (
                f'({self.min_wavelength}, {self.max_wavelength}) µm'
            )
            message += f' within wavelength range {wavelength_range}'

        raise ValueError(message)

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
            self._raise_material_error(no_matches=True)

        if len(filtered_df) > 1 and not self.robust:
            self._raise_material_error(multiple_matches=True)

        material_data = filtered_df.loc[0].to_dict()
        filename = filtered_df.loc[0, 'filename']

        current_dir = os.path.dirname(os.path.abspath(__file__))
        database_dir = os.path.join(current_dir, '../database/data-nk')
        full_filename = os.path.join(database_dir, filename)
        return full_filename, material_data
