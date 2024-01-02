import numpy as np


class BaseRays:

    def translate(self, dx: float, dy: float, dz: float):
        """Shift rays in x, y, z"""
        self.x += dx
        self.y += dy
        self.z += dz

    def _process_input(self, data):
        if isinstance(data, (int, float)):
            return np.array([data], dtype=float)
        elif isinstance(data, np.ndarray):
            return np.ravel(data).astype(float)
        else:
            raise ValueError('Unsupported input type. Must be a scalar '
                             'or a NumPy array.')


class RealRays(BaseRays):

    def __init__(self, x, y, z, L, M, N, energy, wavelength):
        self.x = self._process_input(x)
        self.y = self._process_input(y)
        self.z = self._process_input(z)
        self.L = self._process_input(L)
        self.M = self._process_input(M)
        self.N = self._process_input(N)
        self.e = self._process_input(energy)
        self.w = self._process_input(wavelength)
        self.opd = np.zeros_like(x)

    def rotate_x(self, rx: float):
        """Rotate about x-axis"""
        y = self.y * np.cos(rx) - self.z * np.sin(rx)
        z = self.y * np.sin(rx) + self.z * np.cos(rx)
        m = self.M * np.cos(rx) - self.N * np.sin(rx)
        n = self.M * np.sin(rx) + self.N * np.cos(rx)
        self.y = y
        self.z = z
        self.M = m
        self.N = n

    def rotate_y(self, ry: float):
        """Rotate about y-axis"""
        x = self.x * np.cos(ry) + self.z * np.sin(ry)
        z = -self.x * np.sin(ry) + self.z * np.cos(ry)
        L = self.L * np.cos(ry) + self.N * np.sin(ry)
        n = -self.L * np.sin(ry) + self.N * np.cos(ry)
        self.x = x
        self.z = z
        self.L = L
        self.N = n

    def rotate_z(self, rz: float):
        """Rotate about z-axis"""
        x = self.x * np.cos(rz) - self.y * np.sin(rz)
        y = self.x * np.sin(rz) + self.y * np.cos(rz)
        L = self.L * np.cos(rz) - self.M * np.sin(rz)
        m = self.L * np.sin(rz) + self.M * np.cos(rz)
        self.x = x
        self.y = y
        self.L = L
        self.M = m

    def propagate(self, t: float):
        """Propagate rays a distance t"""
        self.x += t * self.L
        self.y += t * self.M
        self.z += t * self.N

    def clip(self, condition):
        self.e[condition] = 0.0


class ParaxialRays(BaseRays):

    def __init__(self, y, u, z, wavelength):
        self.y = self._process_input(y)
        self.z = self._process_input(z)
        self.u = self._process_input(u)
        self.x = np.zeros_like(y)
        self.e = np.ones_like(y)
        self.w = self._process_input(wavelength)

    def propagate(self, t: float):
        self.z += t
        self.y += t * self.u
