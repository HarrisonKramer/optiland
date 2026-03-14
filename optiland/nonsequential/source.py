"""Non-sequential light sources.

Provides base and concrete source classes for generating rays in
non-sequential scenes.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import optiland.backend as be
from optiland.nonsequential.ray_data import NSQRayPool


class BaseSource(ABC):
    """Abstract base for non-sequential light sources."""

    @abstractmethod
    def generate_rays(self, n_rays: int) -> NSQRayPool:
        """Generate a batch of rays.

        Args:
            n_rays: Number of rays to generate.

        Returns:
            A ray pool containing the generated rays.
        """


class PointSource(BaseSource):
    """Isotropic or directed point source.

    Emits rays from a single point with configurable angular distribution.

    Args:
        position: Source position in global coordinates (x, y, z).
        direction: Central emission direction (x, y, z). Will be normalized.
        half_angle: Half-angle of the emission cone in radians.
            0 = collimated, pi = full hemisphere.
        wavelength: Emission wavelength in micrometers.
        record_paths: Whether generated rays should record path history.
    """

    def __init__(
        self,
        position: tuple[float, float, float] = (0, 0, 0),
        direction: tuple[float, float, float] = (0, 0, 1),
        half_angle: float = 0.0,
        wavelength: float = 0.55,
        record_paths: bool = False,
    ):
        self.position = position
        # Normalize direction
        d = np.array(direction, dtype=float)
        d = d / np.linalg.norm(d)
        self.direction = tuple(d)
        self.half_angle = half_angle
        self.wavelength = wavelength
        self.record_paths = record_paths

    def generate_rays(self, n_rays: int) -> NSQRayPool:
        """Generate rays with uniform random distribution within the cone.

        If half_angle == 0: all rays go in ``direction`` (collimated).
        Otherwise rays are uniformly distributed in solid angle within
        the cone.

        Args:
            n_rays: Number of rays to generate.

        Returns:
            A ray pool containing the generated rays.
        """
        x = be.full(n_rays, float(self.position[0]))
        y = be.full(n_rays, float(self.position[1]))
        z = be.full(n_rays, float(self.position[2]))

        if self.half_angle == 0.0:
            L = be.full(n_rays, float(self.direction[0]))
            M = be.full(n_rays, float(self.direction[1]))
            N = be.full(n_rays, float(self.direction[2]))
        else:
            L, M, N = self._generate_cone_directions(n_rays)

        intensity = be.ones(n_rays)
        wavelength = be.full(n_rays, float(self.wavelength))

        return NSQRayPool(
            x,
            y,
            z,
            L,
            M,
            N,
            intensity,
            wavelength,
            record_paths=self.record_paths,
        )

    def _generate_cone_directions(
        self,
        n_rays: int,
    ) -> tuple:
        """Generate uniformly distributed directions within a cone.

        Uses rejection-free sampling: uniform in cos(theta) within the
        cone, uniform in phi around the axis.

        Args:
            n_rays: Number of directions to generate.

        Returns:
            L, M, N: Direction cosine arrays.
        """
        # Sample uniform in cos(theta) from [cos(half_angle), 1]
        cos_min = np.cos(self.half_angle)
        cos_theta = np.random.uniform(cos_min, 1.0, size=n_rays)
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = np.random.uniform(0, 2 * np.pi, size=n_rays)

        # Directions in local frame (z-axis = cone axis)
        lx = sin_theta * np.cos(phi)
        ly = sin_theta * np.sin(phi)
        lz = cos_theta

        # Rotate from z-axis to self.direction
        dz = np.array(self.direction)
        if np.allclose(dz, [0, 0, 1]):
            L_out, M_out, N_out = lx, ly, lz
        elif np.allclose(dz, [0, 0, -1]):
            L_out, M_out, N_out = lx, -ly, -lz
        else:
            # Build orthonormal basis
            up = np.array([0, 0, 1.0])
            dx = np.cross(up, dz)
            dx = dx / np.linalg.norm(dx)
            dy = np.cross(dz, dx)

            L_out = lx * dx[0] + ly * dy[0] + lz * dz[0]
            M_out = lx * dx[1] + ly * dy[1] + lz * dz[1]
            N_out = lx * dx[2] + ly * dy[2] + lz * dz[2]

        return (
            be.array(L_out),
            be.array(M_out),
            be.array(N_out),
        )
