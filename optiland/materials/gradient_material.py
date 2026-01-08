"""Gradient-Index (GRIN) Material.

This module defines the GradientMaterial class for materials with a spatially
varying refractive index.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.materials.base import BaseMaterial
from optiland.propagation.grin import GRINPropagation

if TYPE_CHECKING:
    from optiland._types import ScalarOrArray


class GradientMaterial(BaseMaterial):
    """Represents a gradient-index material.

    This material supports a refractive index that varies with position,
    following a form similar to Zemax's Gradient3 definition:
        n(r, z) = n0 + nr2*r^2 + nr4*r^4 + nr6*r^6 + nz1*z + nz2*z^2 + nz3*z^3

    where r^2 = x^2 + y^2.

    Args:
        n0: Base refractive index.
        nr2: Coefficient for r^2 term in radial gradient.
        nr4: Coefficient for r^4 term in radial gradient.
        nr6: Coefficient for r^6 term in radial gradient.
        nz1: Coefficient for z term in axial gradient.
        nz2: Coefficient for z^2 term in axial gradient.
        nz3: Coefficient for z^3 term in axial gradient.

    """

    def __init__(
        self,
        n0: float = 1.5,
        nr2: float = 0.0,
        nr4: float = 0.0,
        nr6: float = 0.0,
        nz1: float = 0.0,
        nz2: float = 0.0,
        nz3: float = 0.0,
    ):
        """Initializes the gradient-index material."""
        # Initialize with GRIN propagation model
        super().__init__(propagation_model=None)
        self.propagation_model = GRINPropagation(self)

        # Store gradient coefficients
        self.n0 = n0
        self.nr2 = nr2
        self.nr4 = nr4
        self.nr6 = nr6
        self.nz1 = nz1
        self.nz2 = nz2
        self.nz3 = nz3

    def _calculate_n(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Calculates the refractive index at a given position.

        Args:
            wavelength: The wavelength(s) of light in microns.
            **kwargs: Additional arguments, including:
                - x: x-coordinate(s)
                - y: y-coordinate(s)
                - z: z-coordinate(s)

        Returns:
            The refractive index at the given position(s).

        """
        # Get position if provided
        x = kwargs.get("x", 0.0)
        y = kwargs.get("y", 0.0)
        z = kwargs.get("z", 0.0)

        # Calculate r^2 = x^2 + y^2
        r_sq = x**2 + y**2

        # Calculate refractive index using gradient formula
        n = (
            self.n0
            + self.nr2 * r_sq
            + self.nr4 * r_sq**2
            + self.nr6 * r_sq**3
            + self.nz1 * z
            + self.nz2 * z**2
            + self.nz3 * z**3
        )

        return n

    def _calculate_k(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Calculates the extinction coefficient (assumed to be zero).

        Args:
            wavelength: The wavelength(s) of light in microns.
            **kwargs: Additional arguments (ignored).

        Returns:
            Zero extinction coefficient.

        """
        return be.zeros_like(wavelength) if be.is_array_like(wavelength) else 0.0

    def get_index_and_gradient(
        self, x: ScalarOrArray, y: ScalarOrArray, z: ScalarOrArray, wavelength: float
    ) -> tuple[be.ndarray, be.ndarray, be.ndarray, be.ndarray]:
        """Calculates refractive index and its gradient at a given position.

        Args:
            x: x-coordinate(s).
            y: y-coordinate(s).
            z: z-coordinate(s).
            wavelength: Wavelength in microns (currently not used, n is
                wavelength-independent in this model).

        Returns:
            A tuple containing:
                - n: Refractive index at the position(s).
                - dn_dx: Partial derivative of n with respect to x.
                - dn_dy: Partial derivative of n with respect to y.
                - dn_dz: Partial derivative of n with respect to z.

        """
        x = be.atleast_1d(be.array(x))
        y = be.atleast_1d(be.array(y))
        z = be.atleast_1d(be.array(z))

        # Calculate r^2 = x^2 + y^2
        r_sq = x**2 + y**2

        # Calculate refractive index
        n = (
            self.n0
            + self.nr2 * r_sq
            + self.nr4 * r_sq**2
            + self.nr6 * r_sq**3
            + self.nz1 * z
            + self.nz2 * z**2
            + self.nz3 * z**3
        )

        # Calculate gradient
        # dn/dx = 2*nr2*x + 4*nr4*r_sq*x + 6*nr6*r_sq^2*x
        #        = x * (2*nr2 + 4*nr4*r_sq + 6*nr6*r_sq^2)
        g_r = 2 * self.nr2 + 4 * self.nr4 * r_sq + 6 * self.nr6 * r_sq**2
        dn_dx = x * g_r
        dn_dy = y * g_r

        # dn/dz = nz1 + 2*nz2*z + 3*nz3*z^2
        dn_dz = self.nz1 + 2 * self.nz2 * z + 3 * self.nz3 * z**2

        return n, dn_dx, dn_dy, dn_dz

    def to_dict(self) -> dict:
        """Converts the material to a dictionary.

        Returns:
            Dictionary representation of the material.

        """
        data = super().to_dict()
        data.update({
            "n0": self.n0,
            "nr2": self.nr2,
            "nr4": self.nr4,
            "nr6": self.nr6,
            "nz1": self.nz1,
            "nz2": self.nz2,
            "nz3": self.nz3,
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "GradientMaterial":
        """Creates a GradientMaterial from a dictionary.

        Args:
            data: Dictionary containing the material data.

        Returns:
            GradientMaterial instance.

        """
        return cls(
            n0=data.get("n0", 1.5),
            nr2=data.get("nr2", 0.0),
            nr4=data.get("nr4", 0.0),
            nr6=data.get("nr6", 0.0),
            nz1=data.get("nz1", 0.0),
            nz2=data.get("nz2", 0.0),
            nz3=data.get("nz3", 0.0),
        )
