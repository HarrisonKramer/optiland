import abc
import glob
import os
from io import StringIO
import yaml
import numpy as np


class BaseMaterial:

    def __init__(self, name=''):
        self.name = name

    @abc.abstractmethod
    def n(self, wavelength):
        return

    @abc.abstractmethod
    def k(self, wavelength):
        return

    def abbe(self):
        nD = self.n(0.5893)
        nF = self.n(0.4861)
        nC = self.n(0.6563)
        return (nD - 1) / (nF - nC)


class IdealMaterial(BaseMaterial):

    def __init__(self, n, k, name=''):
        super().__init__(name)
        self.index = n
        self.absorp = k

    def n(self, wavelength):
        return self.index

    def k(self, wavelength):
        return self.absorp


class Mirror(IdealMaterial):

    def __init__(self, name=''):
        super().__init__(n=-1.0, k=0.0, name=name)


class Material(BaseMaterial):

    def __init__(self, name, manufacturer=None):
        super().__init__(name)
        self.manufacturer = manufacturer

        self.files = []
        self.types = []
        self.coeffs = []

        self._k_wavelength = None
        self._k = None

        self._retrieve_file()
        self._read_yaml()
        self._decipher_type()
        self._get_coeffs()
        self._get_extinction_coeffs()

    def n(self, wavelength):
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
        try:
            return np.interp(wavelength, self._k_wavelength, self._k)
        except ValueError:
            raise ValueError('No extinction coefficient data found.')

    def _retrieve_file(self):
        if self.manufacturer:
            database_path = f'database/**/{self.manufacturer}/{self.name}.yml'
        else:
            database_path = f'database/**/{self.name}.yml'

        search_path = os.path.join(os.path.dirname(__file__),
                                   '..', database_path)

        for filename in glob.iglob(search_path, recursive=True):
            self.files.append(filename)

        if not self.files:
            raise ValueError(f'No glass data found for "{self.name}"')

        if len(self.files) > 1:
            error_str = f'''More than one material manufacturer found for
            {self.name}: {self.files}. Please additionally list manufacturer.
            '''
            raise ValueError(error_str)

    def _read_yaml(self):
        with open(self.files[0], 'r') as stream:
            self.data = yaml.safe_load(stream)

        for each in self.data['DATA']:
            if each['type'] is not None:
                self.types.append(each['type'])

    def _decipher_type(self):
        for each in self.data['DATA']:
            if each['type'] is not None:
                self.types.append(each['type'])

    def _get_coeffs(self):
        self.coeffs = [float(k) for k in
                       self.data['DATA'][0]['coefficients'].split()]

    def _get_extinction_coeffs(self):
        if self.data['DATA'][1]['type'] == 'tabulated k':
            data_file = StringIO(self.data['DATA'][1]['data'])
            k_data = np.loadtxt(data_file)
            self._k_wavelength = k_data[:, 0]
            self._k = k_data[:, 1]
