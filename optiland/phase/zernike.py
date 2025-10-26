"""Zernike polynomial phase profile

This module defines the `ZernikePhaseProfile` class, which applies a
Zernike polynomial map as a phase profile over a circular aperture.

Kramer Harrison, 2025
"""

from __future__ import annotations

import typing

from optiland import backend as be
from optiland.phase.base import BasePhaseProfile
from optiland.zernike import ZernikeFringe, ZernikeNoll, ZernikeStandard

if typing.TYPE_CHECKING:
    from optiland._types import ZernikeType
    from optiland.zernike.base import BaseZernike


_ZERNIKE_TYPES: dict[ZernikeType, type[BaseZernike]] = {
    "standard": ZernikeStandard,
    "noll": ZernikeNoll,
    "fringe": ZernikeFringe,
}


class ZernikePhaseProfile(BasePhaseProfile):
    """
    Applies a Zernike polynomial map as a phase profile.

    This class uses a provided Zernike polynomial object (conforming to
    `BaseZernike`) to calculate the phase and its gradient over a
    circular aperture defined by `norm_radius`.

    Args:
        coefficients (list or be.ndarray, optional): The coefficients of the
            Zernike polynomial surface. Defaults to an empty list, indicating
            no Zernike polynomial coefficients are used.
        zernike_type (str, optional): The type of Zernike polynomial to use.
            Defaults to "standard". Options are "standard", "noll", or "fringe".
        norm_radius (float, optional): The normalization radius for the
            Zernike polynomial coordinates. Defaults to 1.
    """

    phase_type = "zernike"

    def __init__(
        self,
        coefficients: be.Array | None = None,
        zernike_type: ZernikeType = "standard",
        norm_radius: float = 1.0,
    ):
        super().__init__()
        if zernike_type not in _ZERNIKE_TYPES:
            raise ValueError(
                "Zernike type must be one of 'standard', 'noll', or 'fringe', got "
                f"{zernike_type}",
            )
        if norm_radius <= 0:
            raise ValueError(
                f"Normalization radius must be positive, got {norm_radius}"
            )

        coefficients = be.atleast_1d(coefficients if coefficients is not None else [])

        self.zernike = _ZERNIKE_TYPES[zernike_type](coeffs=coefficients)
        self.zernike_type: ZernikeType = zernike_type
        self.norm_radius = norm_radius

    @property
    def coefficients(self) -> be.Array:
        """Get the coefficients of the Zernike polynomial surface."""
        return self.zernike.coeffs

    @coefficients.setter
    def coefficients(self, value: be.Array) -> None:
        """Set the coefficients of the Zernike polynomial surface."""
        self.zernike = _ZERNIKE_TYPES[self.zernike_type](coeffs=be.atleast_1d(value))

    def _get_polar_coords(
        self, x: be.Array, y: be.Array
    ) -> tuple[be.Array, be.Array, be.Array]:
        """
        Converts Cartesian coordinates to normalized polar coordinates.

        Args:
            x: The x-coordinates.
            y: The y-coordinates.

        Returns:
            A tuple of (r_cart, r_norm, phi):
            - r_cart: The unnormalized radial coordinate (sqrt(x^2 + y^2)).
            - r_norm: The radial coordinate normalized by `self.norm_radius`.
            - phi: The azimuthal angle (arctan2(y, x)).
        """
        r_cart = be.sqrt(x**2 + y**2)
        r_norm = r_cart / self.norm_radius
        phi = be.arctan2(y, x)
        return r_cart, r_norm, phi

    def get_phase(self, x: be.Array, y: be.Array) -> be.Array:
        """
        Calculates the phase added by the Zernike profile at (x, y).

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.

        Returns:
            The phase at each (x, y) coordinate, calculated by summing all
            Zernike terms.
        """
        _r_cart, r_norm, phi = self._get_polar_coords(x, y)
        return self.zernike.poly(r_norm, phi)

    def get_gradient(self, x: be.Array, y: be.Array) -> tuple[be.Array, be.Array]:
        """
        Calculates the gradient of the phase at coordinates (x, y).

        This method computes the total gradient (d_phi/dx, d_phi/dy) by
        summing the Cartesian gradients of each individual Zernike term.
        It uses the chain rule to convert from polar derivatives
        (dZ/dr_norm, dZ/dphi) to Cartesian derivatives (dZ/dx, dZ/dy).

        dZ/dx = (dZ/dr_norm) * (dr_norm/dx) + (dZ/dphi) * (dphi/dx)
        dZ/dy = (dZ/dr_norm) * (dr_norm/dy) + (dZ/dphi) * (dphi/dy)

        where:
        dr_norm/dx = x / (r_cart * R) = cos(phi) / R
        dr_norm/dy = y / (r_cart * R) = sin(phi) / R
        dphi/dx = -y / r_cart^2 = -sin(phi) / (r_norm * R)
        dphi/dy = x / r_cart^2 = cos(phi) / (r_norm * R)
        R = self.norm_radius

        This simplifies to:
        dZ/dx = (1/R) * [ (dZ/dr_norm) * cos(phi) - (dZ/dphi / r_norm) * sin(phi) ]
        dZ/dy = (1/R) * [ (dZ/dr_norm) * sin(phi) + (dZ/dphi / r_norm) * cos(phi) ]

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.

        Returns:
            A tuple containing the x and y components of the phase gradient
            (d_phi/dx, d_phi/dy).
        """
        r_cart, r_norm, phi = self._get_polar_coords(x, y)

        total_dz_dx = be.zeros_like(x)
        total_dz_dy = be.zeros_like(x)

        # Pre-calculate trigonometric values and safe inverses for the origin
        cos_phi = be.safe_divide(x, r_cart, 1.0)
        sin_phi = be.safe_divide(y, r_cart, 0.0)
        # dZ/dphi is proportional to r_norm^|m|.
        # (dZ/dphi) / r_norm is proportional to r_norm^(|m|-1).
        # This is finite at the origin for |m| >= 1.
        # If m = 0, dZ/dphi = 0, so the term is zero.
        # Thus, safe_divide with 0.0 fallback is correct.
        inv_r_norm = be.safe_divide(1.0, r_norm, 0.0)

        for coeff, (n, m) in zip(
            self.zernike.coeffs, self.zernike.indices, strict=False
        ):
            if coeff == 0:
                continue

            # Get polar derivatives for the (n, m) term:
            # (d(Z_nm)/dr_norm, d(Z_nm)/dphi)
            # Note: get_derivative returns un-normalized, un-coefficient-scaled
            # derivatives of the base polynomial terms.
            (
                dZ_nm_dr_norm,
                dZ_nm_dphi,
            ) = self.zernike.get_derivative(n, m, r_norm, phi)

            # Apply coefficient
            dZ_nm_dr_norm *= coeff
            dZ_nm_dphi *= coeff

            # Apply chain rule and accumulate
            term_dx = (dZ_nm_dr_norm * cos_phi) - (dZ_nm_dphi * inv_r_norm * sin_phi)
            term_dy = (dZ_nm_dr_norm * sin_phi) + (dZ_nm_dphi * inv_r_norm * cos_phi)

            total_dz_dx += term_dx
            total_dz_dy += term_dy

        # Final scaling by 1/Norm Radius
        return total_dz_dx / self.norm_radius, total_dz_dy / self.norm_radius

    def get_paraxial_gradient(self, y: be.Array) -> be.Array:
        """
        Calculates the paraxial phase gradient at y-coordinate.

        This is the gradient d_phi/dy evaluated at x=0.

        Args:
            y: The y-coordinates of the points of interest.

        Returns:
            The paraxial phase gradient (d_phi/dy) at each y-coordinate.
        """
        _dx, dy = self.get_gradient(be.zeros_like(y), y)
        return dy

    def to_dict(self) -> dict:
        """
        Serializes the phase profile to a dictionary.

        Returns:
            A dictionary representation of the phase profile, including
            the normalized radius and Zernike coefficients.
        """
        data = super().to_dict()
        data["norm_radius"] = self.norm_radius
        # Convert backend array to a serializable list
        data["coefficients"] = be.to_list(self.zernike.coeffs)
        # Store the class name of the zernike object to allow reconstruction
        data["zernike_type"] = self.zernike_type
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ZernikePhaseProfile:
        """
        Deserializes a phase profile from a dictionary.

        Note: This implementation assumes that the Zernike class
        (e.g., `ZernikeStandard`) is importable from
        `optiland.zernike.standard`.

        Args:
            data: A dictionary representation of a `ZernikePhaseProfile`.

        Returns:
            An instance of `ZernikePhaseProfile`.

        Raises:
            ImportError: If the required Zernike class cannot be imported.
        """
        coefficients = be.array(data.get("coefficients", []))
        norm_radius = data.get("norm_radius", 1.0)
        zernike_type = data.get("zernike_type", "standard")
        return cls(
            coefficients=coefficients,
            norm_radius=norm_radius,
            zernike_type=zernike_type,
        )
