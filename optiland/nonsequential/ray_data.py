"""Non-sequential ray pool.

Manages all active rays as parallel arrays with boolean masking for
active/inactive rays rather than dynamic resizing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from optiland._types import BEArray


class NSQRayPool:
    """Pool of rays for non-sequential tracing.

    Rays are stored as parallel arrays. Inactive rays are masked
    rather than removed, enabling fixed-size tensor operations.

    Args:
        x: x-coordinates of ray positions.
        y: y-coordinates of ray positions.
        z: z-coordinates of ray positions.
        L: x-direction cosines.
        M: y-direction cosines.
        N: z-direction cosines.
        intensity: Ray intensities.
        wavelength: Per-ray wavelength values.
        record_paths: Whether to record path history.
    """

    def __init__(
        self,
        x: BEArray,
        y: BEArray,
        z: BEArray,
        L: BEArray,
        M: BEArray,
        N: BEArray,
        intensity: BEArray,
        wavelength: BEArray,
        record_paths: bool = False,
    ):
        self.x = be.atleast_1d(be.array(x))
        self.y = be.atleast_1d(be.array(y))
        self.z = be.atleast_1d(be.array(z))
        self.L = be.atleast_1d(be.array(L))
        self.M = be.atleast_1d(be.array(M))
        self.N = be.atleast_1d(be.array(N))
        self.intensity = be.atleast_1d(be.array(intensity))
        self.wavelength = be.atleast_1d(be.array(wavelength))

        n = len(self.x)
        self.active = be.ones(n) > 0  # boolean array
        self.last_surface_id = be.full(n, -1.0)

        self.record_paths = record_paths
        self.path_history: list[dict] | None = [] if record_paths else None

    @property
    def n_active(self) -> int:
        """Number of currently active rays."""
        return int(be.sum(self.active))

    def deactivate(self, mask: BEArray) -> None:
        """Deactivate rays matching the mask.

        Args:
            mask: Boolean array; True = deactivate that ray.
        """
        self.active = self.active & ~mask

    def propagate(self, distances: BEArray) -> None:
        """Advance active rays by given distances along their directions.

        Args:
            distances: Propagation distance for each ray.
        """
        active = self.active
        self.x = be.where(active, self.x + distances * self.L, self.x)
        self.y = be.where(active, self.y + distances * self.M, self.y)
        self.z = be.where(active, self.z + distances * self.N, self.z)

    def record_path_point(self, surface_ids: BEArray) -> None:
        """If path recording is enabled, snapshot current state.

        Args:
            surface_ids: Surface ID hit by each ray at this point.
        """
        if self.path_history is None:
            return
        self.path_history.append(
            {
                "x": be.copy(self.x),
                "y": be.copy(self.y),
                "z": be.copy(self.z),
                "L": be.copy(self.L),
                "M": be.copy(self.M),
                "N": be.copy(self.N),
                "intensity": be.copy(self.intensity),
                "surface_ids": be.copy(surface_ids),
                "active": be.copy(self.active),
            }
        )

    def update_directions(
        self,
        L: BEArray,
        M: BEArray,
        N: BEArray,
        mask: BEArray,
    ) -> None:
        """Update direction cosines for rays matching mask.

        Args:
            L: New x-direction cosines.
            M: New y-direction cosines.
            N: New z-direction cosines.
            mask: Boolean mask selecting which rays to update.
        """
        self.L = be.where(mask, L, self.L)
        self.M = be.where(mask, M, self.M)
        self.N = be.where(mask, N, self.N)

    def apply_intensity(self, factors: BEArray, mask: BEArray) -> None:
        """Multiply intensities by factors for rays matching mask.

        Args:
            factors: Multiplicative intensity factors.
            mask: Boolean mask selecting which rays to update.
        """
        self.intensity = be.where(
            mask,
            self.intensity * factors,
            self.intensity,
        )
