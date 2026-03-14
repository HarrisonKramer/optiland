"""Detector data accumulator for non-sequential ray tracing.

Accumulates ray hit data on detector surfaces across multiple trace calls.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from optiland._types import BEArray
    from optiland.nonsequential.ray_data import NSQRayPool


class DetectorData:
    """Accumulates ray hit data on a detector surface.

    Designed for iterative accumulation: the user calls trace() multiple
    times and detector data accumulates across calls.
    """

    def __init__(self):
        self._x: list[BEArray] = []
        self._y: list[BEArray] = []
        self._z: list[BEArray] = []
        self._L: list[BEArray] = []
        self._M: list[BEArray] = []
        self._N: list[BEArray] = []
        self._intensity: list[BEArray] = []
        self._wavelength: list[BEArray] = []

    def record(self, rays: NSQRayPool, mask: BEArray) -> None:
        """Record rays matching the mask.

        Args:
            rays: The ray pool.
            mask: Boolean mask selecting which rays to record.
        """
        if not be.any(mask):
            return

        nz = be.nonzero(mask)
        # NumPy returns tuple of arrays, torch returns 2D tensor
        indices = nz[0] if isinstance(nz, tuple) else nz.flatten()
        self._x.append(rays.x[indices])
        self._y.append(rays.y[indices])
        self._z.append(rays.z[indices])
        self._L.append(rays.L[indices])
        self._M.append(rays.M[indices])
        self._N.append(rays.N[indices])
        self._intensity.append(rays.intensity[indices])
        self._wavelength.append(rays.wavelength[indices])

    @property
    def n_hits(self) -> int:
        """Total number of recorded hits across all batches."""
        return sum(len(chunk) for chunk in self._x)

    def get_positions(self) -> tuple[BEArray, BEArray, BEArray]:
        """Concatenate all recorded positions.

        Returns:
            x, y, z: Concatenated position arrays.
        """
        if not self._x:
            empty = be.array([])
            return empty, be.copy(empty), be.copy(empty)
        return (
            be.concatenate(self._x),
            be.concatenate(self._y),
            be.concatenate(self._z),
        )

    def get_directions(self) -> tuple[BEArray, BEArray, BEArray]:
        """Concatenate all recorded directions.

        Returns:
            L, M, N: Concatenated direction cosine arrays.
        """
        if not self._L:
            empty = be.array([])
            return empty, be.copy(empty), be.copy(empty)
        return (
            be.concatenate(self._L),
            be.concatenate(self._M),
            be.concatenate(self._N),
        )

    def get_intensities(self) -> BEArray:
        """Concatenate all recorded intensities."""
        if not self._intensity:
            return be.array([])
        return be.concatenate(self._intensity)

    def get_wavelengths(self) -> BEArray:
        """Concatenate all recorded wavelengths."""
        if not self._wavelength:
            return be.array([])
        return be.concatenate(self._wavelength)

    def reset(self) -> None:
        """Clear all recorded data."""
        self._x.clear()
        self._y.clear()
        self._z.clear()
        self._L.clear()
        self._M.clear()
        self._N.clear()
        self._intensity.clear()
        self._wavelength.clear()
