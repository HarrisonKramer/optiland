import abc
import glob
import os
from io import StringIO
import yaml
import numpy as np


class BaseMaterial:

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

    def __init__(self, n, k):
        self.index = n
        self.absorp = k

    def n(self, wavelength):
        return self.index

    def k(self, wavelength):
        return self.absorp


class Mirror(IdealMaterial):

    def __init__(self):
        super().__init__(n=-1.0, k=0.0)


class MaterialFile(BaseMaterial):

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

    def _read_yaml(self):
        with open(self.file, 'r') as stream:
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
        for data in self.data['DATA']:
            if data['type'] == 'tabulated k':
                data_file = StringIO(self.data['DATA'][1]['data'])
                k_data = np.loadtxt(data_file)
                self._k_wavelength = k_data[:, 0]
                self._k = k_data[:, 1]


class Material(MaterialFile):

    def __init__(self, name, reference=None):
        self.name = name
        self.reference = reference
        file = self._retrieve_file()
        super().__init__(file)

    def _retrieve_file(self):
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
