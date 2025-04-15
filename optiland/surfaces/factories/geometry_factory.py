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
    ChebyshevPolynomialGeometry,
    EvenAsphere,
    OddAsphere,
    Plane,
    PolynomialGeometry,
    StandardGeometry,
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
    """

    radius: float = be.inf
    conic: float = 0.0
    coefficients: list[float] = field(default_factory=list)
    tol: float = 1e-6
    max_iter: int = 100
    norm_x: float = 1.0
    norm_y: float = 1.0
    norm_radius: float = 1.0


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
    "standard": _create_standard,
    "even_asphere": _create_even_asphere,
    "odd_asphere": _create_odd_asphere,
    "polynomial": _create_polynomial,
    "chebyshev": _create_chebyshev,
    "zernike": _create_zernike,
    "paraxial": _create_paraxial,
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
