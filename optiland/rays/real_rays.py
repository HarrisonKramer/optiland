"""Real Rays

This module contains the RealRays class, which represents a collection of real
rays in 3D space.

Kramer Harrison, 2024
"""

import optiland.backend as be
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
        self.opd = be.zeros_like(self.x)

        # variables to hold pre-surface direction cosines
        self.L0 = None
        self.M0 = None
        self.N0 = None

        self.is_normalized = True

    def rotate_x(self, rx: float):
        """Rotate the rays about the x-axis."""
        y = self.y * be.cos(rx) - self.z * be.sin(rx)
        z = self.y * be.sin(rx) + self.z * be.cos(rx)
        m = self.M * be.cos(rx) - self.N * be.sin(rx)
        n = self.M * be.sin(rx) + self.N * be.cos(rx)
        self.y = y
        self.z = z
        self.M = m
        self.N = n

    def rotate_y(self, ry: float):
        """Rotate the rays about the y-axis."""
        x = self.x * be.cos(ry) + self.z * be.sin(ry)
        z = -self.x * be.sin(ry) + self.z * be.cos(ry)
        L = self.L * be.cos(ry) + self.N * be.sin(ry)
        n = -self.L * be.sin(ry) + self.N * be.cos(ry)
        self.x = x
        self.z = z
        self.L = L
        self.N = n

    def rotate_z(self, rz: float):
        """Rotate the rays about the z-axis."""
        x = self.x * be.cos(rz) - self.y * be.sin(rz)
        y = self.x * be.sin(rz) + self.y * be.cos(rz)
        L = self.L * be.cos(rz) - self.M * be.sin(rz)
        m = self.L * be.sin(rz) + self.M * be.cos(rz)
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
            alpha = 4 * be.pi * k / self.w
            self.i *= be.exp(-alpha * t * 1e3)  # mm to microns

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

        root = be.sqrt(1 - u**2 * (1 - dot**2))
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

    def update(self, jones_matrix: be.ndarray = None):
        """Update ray properties (primarily used for polarization)."""

    def normalize(self):
        """Normalize the direction vectors of the rays."""
        mag = be.sqrt(self.L**2 + self.M**2 + self.N**2)
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

        sgn = be.sign(dot)
        nx *= sgn
        ny *= sgn
        nz *= sgn

        dot = be.abs(dot)
        return nx, ny, nz, dot
