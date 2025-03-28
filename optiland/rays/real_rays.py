"""Real Rays

This module contains the RealRays class, which represents a collection of real
rays in 3D space.

Kramer Harrison, 2024
"""

import numpy as np

from optiland.materials import BaseMaterial
from optiland.rays.base import BaseRays


class RealRays(BaseRays):
    """Represents a collection of real rays in 3D space.

    Attributes:
        x (ndarray): The x-coordinates of the rays.
        y (ndarray): The y-coordinates of the rays.
        z (ndarray): The z-coordinates of the rays.
        L (ndarray): The x-components of the direction vectors of the rays.
        M (ndarray): The y-components of the direction vectors of the rays.
        N (ndarray): The z-components of the direction vectors of the rays.
        i (ndarray): The intensity of the rays.
        w (ndarray): The wavelength of the rays.
        opd (ndarray): The optical path length of the rays.

    Methods:
        rotate_x(rx: float): Rotate the rays about the x-axis.
        rotate_y(ry: float): Rotate the rays about the y-axis.
        rotate_z(rz: float): Rotate the rays about the z-axis.
        propagate(t: float): Propagate the rays a distance t.
        clip(condition): Clip the rays based on a condition.

    """

    def __init__(self, x, y, z, L, M, N, intensity, wavelength):
        self.x = self._process_input(x)
        self.y = self._process_input(y)
        self.z = self._process_input(z)
        self.L = self._process_input(L)
        self.M = self._process_input(M)
        self.N = self._process_input(N)
        self.i = self._process_input(intensity)
        self.w = self._process_input(wavelength)
        self.opd = np.zeros_like(self.x)

        # variables to hold pre-surface direction cosines
        self.L0 = None
        self.M0 = None
        self.N0 = None

        self.is_normalized = True

    def rotate_x(self, rx: float):
        """Rotate the rays about the x-axis."""
        y = self.y * np.cos(rx) - self.z * np.sin(rx)
        z = self.y * np.sin(rx) + self.z * np.cos(rx)
        m = self.M * np.cos(rx) - self.N * np.sin(rx)
        n = self.M * np.sin(rx) + self.N * np.cos(rx)
        self.y = y
        self.z = z
        self.M = m
        self.N = n

    def rotate_y(self, ry: float):
        """Rotate the rays about the y-axis."""
        x = self.x * np.cos(ry) + self.z * np.sin(ry)
        z = -self.x * np.sin(ry) + self.z * np.cos(ry)
        L = self.L * np.cos(ry) + self.N * np.sin(ry)
        n = -self.L * np.sin(ry) + self.N * np.cos(ry)
        self.x = x
        self.z = z
        self.L = L
        self.N = n

    def rotate_z(self, rz: float):
        """Rotate the rays about the z-axis."""
        x = self.x * np.cos(rz) - self.y * np.sin(rz)
        y = self.x * np.sin(rz) + self.y * np.cos(rz)
        L = self.L * np.cos(rz) - self.M * np.sin(rz)
        m = self.L * np.sin(rz) + self.M * np.cos(rz)
        self.x = x
        self.y = y
        self.L = L
        self.M = m

    def propagate(self, t: float, material: BaseMaterial = None):
        """Propagate the rays a distance t."""
        self.x += t * self.L
        self.y += t * self.M
        self.z += t * self.N

        if material is not None:
            k = material.k(self.w)
            alpha = 4 * np.pi * k / self.w
            self.i *= np.exp(-alpha * t * 1e3)  # mm to microns

        # normalize, if required
        if not self.is_normalized:
            self.normalize()

    def clip(self, condition):
        """Clip the rays based on a condition."""
        self.i[condition] = 0.0

    def refract(self, nx, ny, nz, n1, n2):
        """Refract rays on the surface.

        Args:
            rays: The rays.
            nx: The x-component of the surface normals.
            ny: The y-component of the surface normals.
            nz: The z-component of the surface normals.

        Returns:
            RealRays: The refracted rays.

        """
        self.L0 = self.L.copy()
        self.M0 = self.M.copy()
        self.N0 = self.N.copy()

        u = n1 / n2
        nx, ny, nz, dot = self._align_surface_normal(nx, ny, nz)

        root = np.sqrt(1 - u**2 * (1 - dot**2))
        tx = u * self.L0 + nx * root - u * nx * dot
        ty = u * self.M0 + ny * root - u * ny * dot
        tz = u * self.N0 + nz * root - u * nz * dot

        self.L = tx
        self.M = ty
        self.N = tz

    def reflect(self, nx, ny, nz):
        """Reflects the rays on the surface.

        Args:
            nx: The x-component of the surface normal.
            ny: The y-component of the surface normal.
            nz: The z-component of the surface normal.

        Returns:
            RealRays: The reflected rays.

        """
        self.L0 = self.L.copy()
        self.M0 = self.M.copy()
        self.N0 = self.N.copy()

        nx, ny, nz, dot = self._align_surface_normal(nx, ny, nz)

        self.L -= 2 * dot * nx
        self.M -= 2 * dot * ny
        self.N -= 2 * dot * nz

    def update(self, jones_matrix: np.ndarray = None):
        """Update ray properties (primarily used for polarization)."""

    def normalize(self):
        """Normalize the direction vectors of the rays."""
        mag = np.sqrt(self.L**2 + self.M**2 + self.N**2)
        self.L /= mag
        self.M /= mag
        self.N /= mag
        self.is_normalized = True

    def _align_surface_normal(self, nx, ny, nz):
        """Align the surface normal with the incident ray vectors.

        Note:
            This is done as a convention to ensure the surface normal is
            pointing in the correct direction. This is required for consistency
            with the vector reflection and refraction equations used.

        Args:
            nx: The x-component of the surface normal.
            ny: The y-component of the surface normal.
            nz: The z-component of the surface normal.

        Returns:
            nx: The corrected x-component of the surface normal.
            ny: The corrected y-component of the surface normal.
            nz: The corrected z-component of the surface normal.
            dot: The dot product of the surface normal and the incident ray
                vectors.

        """
        dot = self.L0 * nx + self.M0 * ny + self.N0 * nz

        sgn = np.sign(dot)
        nx *= sgn
        ny *= sgn
        nz *= sgn

        dot = np.abs(dot)
        return nx, ny, nz, dot
