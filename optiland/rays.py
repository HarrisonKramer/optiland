import numpy as np


class BaseRays:

    def translate(self, dx: float, dy: float, dz: float):
        """Shift rays in x, y, z"""
        self.x += dx
        self.y += dy
        self.z += dz


class RealRays(BaseRays):

    def __init__(self, x, y, z, L, M, N, energy, wavelength):
        self.x = x
        self.y = y
        self.z = z
        self.L = L
        self.M = M
        self.N = N
        self.e = energy
        self.w = wavelength
        self.opd = np.zeros_like(x, dtype=float)

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
        self.x[condition] = np.nan
        self.y[condition] = np.nan
        self.z[condition] = np.nan
        self.L[condition] = np.nan
        self.M[condition] = np.nan
        self.N[condition] = np.nan
        self.e[condition] = np.nan
        self.w[condition] = np.nan


class ParaxialRays(BaseRays):

    def __init__(self, y, u, z, wavelength):
        self.x = np.zeros_like(y)
        self.y = y
        self.z = z
        self.u = u
        self.e = np.ones_like(y)
        self.w = wavelength

    def propagate(self, t: float):
        self.z += t
        self.y += t * self.u

    def clip(self, condition):
        self.y[condition] = np.nan
        self.u[condition] = np.nan
