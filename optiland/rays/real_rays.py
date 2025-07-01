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
        self.x = be.as_array_1d(x)
        self.y = be.as_array_1d(y)
        self.z = be.as_array_1d(z)
        self.L = be.as_array_1d(L)
        self.M = be.as_array_1d(M)
        self.N = be.as_array_1d(N)
        self.i = be.as_array_1d(intensity)
        self.w = be.as_array_1d(wavelength)
        self.opd = be.zeros_like(self.x)

        # variables to hold pre-surface direction cosines
        self.L0 = None
        self.M0 = None
        self.N0 = None

        self.is_normalized = True

    def rotate_x(self, rx: float):
        """Rotate the rays about the x-axis."""
        rx = be.array(rx)
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
        ry = be.array(ry)
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
        rz = be.array(rz)
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
        self.x = self.x + t * self.L
        self.y = self.y + t * self.M
        self.z = self.z + t * self.N

        if material is not None:
            k = material.k(self.w)
            alpha = 4 * be.pi * k / self.w
            self.i = self.i * be.exp(-alpha * t * 1e3)  # mm to microns

        # normalize, if required
        if not self.is_normalized:
            self.normalize()

    def clip(self, condition):
        """Clip the rays based on a condition."""
        cond = be.array(condition)
        try:
            cond = cond.astype(bool)
        except AttributeError:
            cond = cond.bool()
        self.i = be.where(cond, be.zeros_like(self.i), self.i)

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
        self.L0 = be.copy(self.L)
        self.M0 = be.copy(self.M)
        self.N0 = be.copy(self.N)

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
        self.L0 = be.copy(self.L)
        self.M0 = be.copy(self.M)
        self.N0 = be.copy(self.N)

        nx, ny, nz, dot = self._align_surface_normal(nx, ny, nz)

        self.L = self.L - 2 * dot * nx
        self.M = self.M - 2 * dot * ny
        self.N = self.N - 2 * dot * nz

    def update(self, jones_matrix: be.ndarray = None):
        """Update ray properties (primarily used for polarization)."""

    def normalize(self):
        """Normalize the direction vectors of the rays."""
        mag = be.sqrt(self.L**2 + self.M**2 + self.N**2)
        self.L = self.L / mag
        self.M = self.M / mag
        self.N = self.N / mag
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
        nx = nx * sgn
        ny = ny * sgn
        nz = nz * sgn

        dot = be.abs(dot)
        return nx, ny, nz, dot

    def __str__(self):
        """Returns a string representation of the rays in a tabular format.
        Truncates output if the number of rays is large, showing first,
        central, and last rays.
        """

        if self.x is None or len(self.x) == 0:
            return "RealRays object (No rays)"

        num_rays = len(self.x)
        max_rays_to_print = 3
        header = (
            f"{'Ray #':>6} | {'x':>10} | {'y':>10} | {'z':>10} | "
            f"{'L':>10} | {'M':>10} | {'N':>10} | "
            f"{'Intensity':>10} | {'Wavelength':>12}\n"
        )
        separator = "-" * (len(header) + 5) + "\n"

        table = header + separator

        def format_ray(i):
            if 0 <= i < num_rays:
                x = be.to_numpy(self.x)[i]
                y = be.to_numpy(self.y)[i]
                z = be.to_numpy(self.z)[i]
                L = be.to_numpy(self.L)[i]
                M = be.to_numpy(self.M)[i]
                N = be.to_numpy(self.N)[i]
                intensity = be.to_numpy(self.i)[i]
                wavelength = be.to_numpy(self.w)[i]
                txt = (
                    f"{i:6} | {x:10.4f} | {y:10.4f} | {z:10.4f} | {L:10.6f} | "
                    f"{M:10.6f} | {N:10.6f} | "
                    f"{intensity:10.4f} | {wavelength:12.4f}\n"
                )
                return txt
            return ""

        if num_rays <= max_rays_to_print:
            indices_to_print = list(range(num_rays))
            count_shown = num_rays

            for i in indices_to_print:
                table += format_ray(i)

        else:
            num_ends = (max_rays_to_print - 1) // 2
            central_index = num_rays // 2

            indices = (
                set(range(num_ends))
                | {central_index}
                | set(range(num_rays - num_ends, num_rays))
            )

            sorted_indices = sorted(list(indices))
            count_shown = len(sorted_indices)

            for i in sorted_indices:
                table += format_ray(i)

        table += separator
        table += f"Showing {count_shown} of {num_rays} rays.\n"

        return table
