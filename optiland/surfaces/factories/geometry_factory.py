"""Geometry Factory Module

This module contains the GeometryFactory class, which is responsible for generating
an appropriate geometry instance, given an input configuration. The class interfaces
tightly with the surface factory for building surfaces, which are the building
blocks of optical systems in Optiland.

Kramer Harrison, 2025
"""

from dataclasses import dataclass, field
from typing import Any

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import (
    BiconicGeometry,
    ChebyshevPolynomialGeometry,
    EvenAsphere,
    OddAsphere,
    Plane,
    PolynomialGeometry,
    StandardGeometry,
    ToroidalGeometry,
    ZernikePolynomialGeometry,
)


@dataclass
class GeometryConfig:
    """Configuration parameters for creating a surface geometry.

    Attributes:
        radius (float): radius of curvature of the geometry. Defaults to be.inf.
        conic (float): conic constant of the geometry. Defaults to 0.0.as_integer_ratio
        coefficients (list): list of geometry coefficients. Defaults to empty list.
        tol (float): tolerance to use for Newton-Raphson method. Defaults to 1e-6.
        max_iter (int): maximum number of iterations to use for Newton-Raphson method.
            Defaults to 100.
        norm_x (float): normalization factor in x. Defaults to 1.0.
        norm_y (float): normalization factor in y. Defaults to 1.0.
        norm_radius (float): normalization radius. Defaults to 1.0.
        radius_x (float): radius of curvature in x for biconic. Defaults to be.inf.
        radius_y (float): radius of curvature in y for biconic or YZ radius for
            toroidal. Defaults to be.inf.
        conic_x (float): conic constant in x for biconic. Defaults to 0.0.
        conic_y (float): conic constant in y for biconic. Defaults to 0.0.
        toroidal_coeffs_poly_y (list): toroidal YZ polynomial coefficients.
                                    Defaults to empty list.
    """

    radius: float = be.inf
    conic: float = 0.0
    coefficients: list[float] = field(default_factory=list)
    tol: float = 1e-6
    max_iter: int = 100
    norm_x: float = 1.0
    norm_y: float = 1.0
    norm_radius: float = 1.0
    # Biconic and Toroidal parameters
    radius_x: float = be.inf  # Used by Biconic
    radius_y: float = be.inf  # Used by Biconic and Toroidal (as radius_yz)
    conic_x: float = 0.0  # Used by Biconic
    conic_y: float = 0.0  # Used by Biconic
    toroidal_coeffs_poly_y: list[float] = field(default_factory=list)


def _create_plane(cs: CoordinateSystem, config: GeometryConfig):
    """
    Create a planar geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (GeometryConfig): configuration of the geometry.

    Returns:
        Plane
    """
    return Plane(cs)


def _create_standard(cs: CoordinateSystem, config: GeometryConfig):
    """
    Create a standard geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (GeometryConfig): configuration of the geometry.

    Returns:
        StandardGeometry or Plane
    """
    # Use a Plane if the radius is infinity.
    if be.isinf(config.radius):
        return Plane(cs)
    return StandardGeometry(cs, config.radius, config.conic)


def _create_even_asphere(cs: CoordinateSystem, config: GeometryConfig):
    """
    Create an even asphere geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (GeometryConfig): configuration of the geometry.

    Returns:
        EvenAsphere
    """
    return EvenAsphere(
        cs,
        config.radius,
        config.conic,
        config.tol,
        config.max_iter,
        config.coefficients,
    )


def _create_odd_asphere(cs: CoordinateSystem, config: GeometryConfig):
    """
    Create an odd asphere geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (GeometryConfig): configuration of the geometry.

    Returns:
        OddAsphere
    """
    return OddAsphere(
        cs,
        config.radius,
        config.conic,
        config.tol,
        config.max_iter,
        config.coefficients,
    )


def _create_polynomial(cs: CoordinateSystem, config: GeometryConfig):
    """
    Create a polynomial geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (GeometryConfig): configuration of the geometry.

    Returns:
        PolynomialGeometry
    """
    return PolynomialGeometry(
        cs,
        config.radius,
        config.conic,
        config.tol,
        config.max_iter,
        config.coefficients,
    )


def _create_chebyshev(cs: CoordinateSystem, config: GeometryConfig):
    """
    Create a Chebyshev geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (GeometryConfig): configuration of the geometry.

    Returns:
        ChebyshevPolynomialGeometry
    """
    return ChebyshevPolynomialGeometry(
        cs,
        config.radius,
        config.conic,
        config.tol,
        config.max_iter,
        config.coefficients,
        config.norm_x,
        config.norm_y,
    )


def _create_zernike(cs: CoordinateSystem, config: GeometryConfig):
    """
    Create a Zernike geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (GeometryConfig): configuration of the geometry.

    Returns:
        ZernikePolynomialGeometry
    """
    return ZernikePolynomialGeometry(
        cs,
        config.radius,
        config.conic,
        config.tol,
        config.max_iter,
        config.coefficients,
        config.norm_radius,
    )


def _create_biconic(cs: CoordinateSystem, config: GeometryConfig):
    """
    Create a biconic geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (GeometryConfig): configuration of the geometry.

    Returns:
        BiconicGeometry
    """
    if (
        be.isinf(config.radius_x)
        and be.isinf(config.radius_y)
        and config.conic_x == 0.0
        and config.conic_y == 0.0
    ):
        # If all radii are infinite and conics are zero, it's a plane
        return Plane(cs)
    return BiconicGeometry(
        coordinate_system=cs,
        radius_x=config.radius_x,
        radius_y=config.radius_y,
        conic_x=config.conic_x,
        conic_y=config.conic_y,
        tol=config.tol,
        max_iter=config.max_iter,
    )


def _create_toroidal(cs: CoordinateSystem, config: GeometryConfig):
    """
    Create a Toroidal geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (GeometryConfig): configuration of the geometry.

    Returns:
        ToroidalGeometry
    """

    return ToroidalGeometry(
        coordinate_system=cs,
        radius_rotation=config.radius,  # Toroidal uses the main 'radius' for rotation
        radius_yz=config.radius_y,  # Toroidal uses 'radius_y' for its YZ radius
        conic=config.conic,
        coeffs_poly_y=config.toroidal_coeffs_poly_y,
        tol=config.tol,
        max_iter=config.max_iter,
    )


def _create_paraxial(cs: CoordinateSystem, config: GeometryConfig):
    """
    Create a paraxial geometry, which is simply a planar surface.

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (GeometryConfig): configuration of the geometry.

    Returns:
        Plane
    """
    return _create_plane(cs, config)


geometry_mapper = {
    "biconic": _create_biconic,
    "chebyshev": _create_chebyshev,
    "even_asphere": _create_even_asphere,
    "odd_asphere": _create_odd_asphere,
    "paraxial": _create_paraxial,
    "polynomial": _create_polynomial,
    "standard": _create_standard,
    "toroidal": _create_toroidal,
    "zernike": _create_zernike,
}


class GeometryFactory:
    """Factory for creating surface geometry objects based on configuration."""

    @staticmethod
    def create(surface_type: str, cs: Any, config: GeometryConfig) -> Any:
        """
        Create and return a geometry object based on the surface type and configuration.

        Args:
            surface_type (str): The type of surface (e.g., 'standard', 'even_asphere').
            cs: The coordinate system for the geometry.
            config (GeometryConfig): Configuration parameters for the geometry.

        Returns:
            The constructed geometry object.

        Raises:
            ValueError: If the surface type is not recognized.
        """
        try:
            create_fn = geometry_mapper[surface_type]
        except KeyError as err:
            raise ValueError(f"Surface type '{surface_type}' not recognized.") from err
        return create_fn(cs, config)
