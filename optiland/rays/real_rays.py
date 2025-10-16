"""Real Rays

This module contains the RealRays class, which represents a collection of real
rays in 3D space. The complex diffraction logic has been removed to adhere to
the Single Responsibility Principle.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.rays.base import BaseRays

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from optiland._types import BEArray, ScalarOrArray


class RealRays(BaseRays):
    """Represents a collection of real rays in 3D space.

    This class stores ray positions, directions, and properties as arrays.
    It is responsible for managing the state of the rays but does not contain
    complex physics for surface interactions.
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        L: ArrayLike,
        M: ArrayLike,
        N: ArrayLike,
        intensity: ArrayLike,
        wavelength: ArrayLike,
    ):
        """Initialize a collection of real rays in 3D space."""
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
        self.L0: BEArray | None = None
        self.M0: BEArray | None = None
        self.N0: BEArray | None = None

        self.is_normalized = True

    def rotate_x(self, rx: ScalarOrArray):
        """Rotate the rays about the x-axis."""
        rx = be.array(rx)
        self.y, self.z, self.M, self.N = (
            self.y * be.cos(rx) - self.z * be.sin(rx),
            self.y * be.sin(rx) + self.z * be.cos(rx),
            self.M * be.cos(rx) - self.N * be.sin(rx),
            self.M * be.sin(rx) + self.N * be.cos(rx),
        )

    def rotate_y(self, ry: ScalarOrArray):
        """Rotate the rays about the y-axis."""
        ry = be.array(ry)
        self.x, self.z, self.L, self.N = (
            self.x * be.cos(ry) + self.z * be.sin(ry),
            -self.x * be.sin(ry) + self.z * be.cos(ry),
            self.L * be.cos(ry) + self.N * be.sin(ry),
            -self.L * be.sin(ry) + self.N * be.cos(ry),
        )

    def rotate_z(self, rz: ScalarOrArray):
        """Rotate the rays about the z-axis."""
        rz = be.array(rz)
        self.x, self.y, self.L, self.M = (
            self.x * be.cos(rz) - self.y * be.sin(rz),
            self.x * be.sin(rz) + self.y * be.cos(rz),
            self.L * be.cos(rz) - self.M * be.sin(rz),
            self.L * be.sin(rz) + self.M * be.cos(rz),
        )

    def clip(self, condition: BEArray):
        """Clip the rays based on a condition (sets intensity to zero)."""
        cond = be.array(condition)
        try:
            cond = cond.astype(bool)
        except AttributeError:
            cond = cond.bool()
        self.i = be.where(cond, be.zeros_like(self.i), self.i)

    def store_incident_direction(self):
        """Stores the current direction cosines before a surface interaction."""
        self.L0 = be.copy(self.L)
        self.M0 = be.copy(self.M)
        self.N0 = be.copy(self.N)

    def refract(self, nx: BEArray, ny: BEArray, nz: BEArray, n1: BEArray, n2: BEArray):
        """Refract rays on the surface using the vector formula."""
        self.store_incident_direction()

        u = n1 / n2
        nx, ny, nz, dot = self._align_surface_normal(nx, ny, nz)

        root = be.sqrt(1 - u**2 * (1 - dot**2))
        self.L = u * self.L0 + nx * root - u * nx * dot
        self.M = u * self.M0 + ny * root - u * ny * dot
        self.N = u * self.N0 + nz * root - u * nz * dot

    def reflect(self, nx: float, ny: float, nz: float):
        """Reflects the rays on the surface.

        Args:
            nx: The x-component of the surface normal.
            ny: The y-component of the surface normal.
            nz: The z-component of the surface normal.
        """
        self.L0 = be.copy(self.L)
        self.M0 = be.copy(self.M)
        self.N0 = be.copy(self.N)

        nx, ny, nz, dot = self._align_surface_normal(nx, ny, nz)

        self.L = self.L - 2 * dot * nx
        self.M = self.M - 2 * dot * ny
        self.N = self.N - 2 * dot * nz

    def update(self, jones_matrix: BEArray | None = None):
        """Update ray properties (primarily used for polarization)."""
        # This method can be expanded for polarization ray tracing.
        pass

    def normalize(self):
        """Normalize the direction vectors of the rays."""
        mag = be.sqrt(self.L**2 + self.M**2 + self.N**2)
        # Avoid division by zero for zero-magnitude vectors
        mag = be.where(mag == 0, be.ones_like(mag), mag)
        self.L = self.L / mag
        self.M = self.M / mag
        self.N = self.N / mag
        self.is_normalized = True

    def _align_surface_normal(
        self, nx: BEArray, ny: BEArray, nz: BEArray
    ) -> tuple[BEArray, BEArray, BEArray, BEArray]:
        """Aligns the surface normal to point away from the incident ray."""
        if self.L0 is None or self.M0 is None or self.N0 is None:
            raise ValueError(
                "Incident direction (L0, M0, N0) must be stored before aligning."
            )
        dot = self.L0 * nx + self.M0 * ny + self.N0 * nz

        # Ensure the normal is pointing against the incident ray
        sgn = be.sign(dot)
        nx = nx * sgn
        ny = ny * sgn
        nz = nz * sgn

        dot = be.abs(dot)
        return nx, ny, nz, dot

    def __str__(self) -> str:
        """Returns a string representation of the rays."""
        if self.x is None or len(self.x) == 0:
            return "RealRays object (No rays)"
        # The detailed string formatting from the original is good and kept here.
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
        else:
            num_ends = (max_rays_to_print - 1) // 2
            indices_to_print = sorted(
                list(
                    set(range(num_ends))
                    | {num_rays // 2}
                    | set(range(num_rays - num_ends, num_rays))
                )
            )
        for i in indices_to_print:
            table += format_ray(i)
        table += separator + f"Showing {len(indices_to_print)} of {num_rays} rays.\n"
        return table
