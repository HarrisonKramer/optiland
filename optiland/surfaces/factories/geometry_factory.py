"""Geometry Factory Module

This module contains the GeometryFactory class, which is responsible for generating
an appropriate geometry instance, given an input configuration. The class interfaces
tightly with the surface factory for building surfaces, which are the building
blocks of optical systems in Optiland.

Kramer Harrison, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.geometries import (
    BiconicGeometry,
    ChebyshevPolynomialGeometry,
    EvenAsphere,
    ForbesQ2dGeometry,
    ForbesQbfsGeometry,
    ForbesSolverConfig,  # forbes
    ForbesSurfaceConfig,  # forbes
    NurbsGeometry,
    OddAsphere,
    Plane,
    PlaneGrating,
    PolynomialGeometry,
    StandardGeometry,
    StandardGratingGeometry,
    ToroidalGeometry,
    ZernikePolynomialGeometry,
)

if TYPE_CHECKING:
    from optiland._types import ZernikeType
    from optiland.coordinate_system import CoordinateSystem


@dataclass
class GeometryConfig:
    """Configuration parameters for creating a surface geometry.

    Attributes:
        radius (float): radius of curvature of the geometry (R = 1/c).
                        Defaults to be.inf.
        conic (float): conic constant (k) of the geometry. Defaults to 0.0.
        grating_order (int): order of the grating. Defaults to 0.
        grating_period (float): period of the grating. Defaults to be.inf.
        groove_orientation_angle (float): angle of the groove orientation.
                                        Defaults to 0.0.
        coefficients (list): list of geometry coefficients. Defaults to empty list.
        tol (float): tolerance to use for Newton-Raphson method. Defaults to 1e-6.
        max_iter (int): maximum number of iterations to use for Newton-Raphson method.
            Defaults to 100.
        norm_x (float): normalization factor in x. Defaults to 1.0.
        norm_y (float): normalization factor in y. Defaults to 1.0.
        norm_radius (float): normalization radius. Defaults to 1.0.
        radius_x (float): radius of curvature in x for biconic.
                          Defaults to be.inf.
        radius_y (float): radius of curvature in y for biconic or YZ radius for
            toroidal. Defaults to be.inf.
        conic_x (float): conic constant in x for biconic. Defaults to 0.0.
        conic_y (float): conic constant in y for biconic. Defaults to 0.0.
        toroidal_coeffs_poly_y (list): toroidal YZ polynomial coefficients.
                                    Defaults to empty list.
        zernike_type (str): type of Zernike polynomial to use. Defaults to "fringe".
        radial_terms (dict): radial terms for Forbes Q-BFS surfaces.
        freeform_coeffs (dict): freeform coefficients for Forbes Q-2D surfaces.
    """

    radius: float = be.inf
    conic: float = 0.0
    grating_order: int = 0
    grating_period: float = be.inf
    groove_orientation_angle: float = 0.0
    coefficients: list[float] = field(default_factory=list)
    tol: float = 1e-6
    max_iter: int = 100
    norm_x: float = 1.0
    norm_y: float = 1.0
    norm_radius: float = 1.0
    # Biconic and Toroidal parameters
    radius_x: float = be.inf  # Used by Biconic and Toroidal (as radius_x)
    radius_y: float = be.inf  # Used by Biconic and Toroidal (as radius_yz)
    conic_x: float = 0.0  # Used by Biconic
    conic_y: float = 0.0  # Used by Biconic
    toroidal_coeffs_poly_y: list[float] = field(default_factory=list)
    zernike_type: ZernikeType = "fringe"
    # Forbes parameters
    radial_terms: dict[int, float] = field(default_factory=dict)
    freeform_coeffs: dict[tuple[str, int, int], float] = field(default_factory=dict)
    # NURBS parameters
    control_points: list[list[list[float]]] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    u_knots: list[float] = field(default_factory=list)
    v_knots: list[float] = field(default_factory=list)
    nurbs_norm_x: float = 0.0
    nurbs_norm_y: float = 0.0
    nurbs_x_center: float = 0.0
    nurbs_y_center: float = 0.0
    u_degree: int = 3
    v_degree: int = 3
    n_points_u: int = 5
    n_points_v: int = 5


def _create_plane(cs: CoordinateSystem, config: GeometryConfig):
    """
    Create a planar geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (GeometryConfig): configuration of the geometry.

    Returns:
        Plane
    """
    return Plane(coordinate_system=cs)


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
    return StandardGeometry(
        coordinate_system=cs, radius=config.radius, conic=config.conic
    )


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
        coordinate_system=cs,
        radius=config.radius,
        conic=config.conic,
        tol=config.tol,
        max_iter=config.max_iter,
        coefficients=config.coefficients,
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
        coordinate_system=cs,
        radius=config.radius,
        conic=config.conic,
        tol=config.tol,
        max_iter=config.max_iter,
        coefficients=config.coefficients,
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
        coordinate_system=cs,
        radius=config.radius,
        conic=config.conic,
        tol=config.tol,
        max_iter=config.max_iter,
        coefficients=config.coefficients,
    )


def _create_grating(cs: CoordinateSystem, config: GeometryConfig):
    """
    Create a grating geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (GeometryConfig): configuration of the geometry.

    Returns:
        StandardGratingGeometry
    """
    # Use a Plane if the radius is infinity.
    if be.isinf(config.radius):
        return PlaneGrating(
            cs,
            config.grating_order,
            config.grating_period,
            config.groove_orientation_angle,
        )
    return StandardGratingGeometry(
        cs,
        config.radius,
        config.grating_order,
        config.grating_period,
        config.groove_orientation_angle,
        config.conic,
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
        coordinate_system=cs,
        radius=config.radius,
        conic=config.conic,
        tol=config.tol,
        max_iter=config.max_iter,
        coefficients=config.coefficients,
        norm_x=config.norm_x,
        norm_y=config.norm_y,
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
        coordinate_system=cs,
        radius=config.radius,
        conic=config.conic,
        tol=config.tol,
        max_iter=config.max_iter,
        zernike_type=config.zernike_type,
        coefficients=config.coefficients,
        norm_radius=config.norm_radius,
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
        radius_x=config.radius_x,  # Toroidal uses radius_x for its X radius
        radius_y=config.radius_y,  # Toroidal uses 'radius_y' for its YZ radius
        conic=config.conic,
        coeffs_poly_y=config.toroidal_coeffs_poly_y,
        tol=config.tol,
        max_iter=config.max_iter,
    )


def _create_forbes_qbfs(cs: CoordinateSystem, config: GeometryConfig):
    """Create a Forbes (Q-BFS) Geometry."""
    surface_config = ForbesSurfaceConfig(
        radius=config.radius,
        conic=config.conic,
        terms=config.radial_terms,
        norm_radius=config.norm_radius,
    )
    solver_config = ForbesSolverConfig(tol=config.tol, max_iter=config.max_iter)

    return ForbesQbfsGeometry(
        cs,
        surface_config=surface_config,
        solver_config=solver_config,
    )


def _create_forbes_q2d(cs: CoordinateSystem, config: GeometryConfig):
    """Create a Forbes (Q-2D) geometry."""
    surface_config = ForbesSurfaceConfig(
        radius=config.radius,
        conic=config.conic,
        terms=config.freeform_coeffs,
        norm_radius=config.norm_radius,
    )
    solver_config = ForbesSolverConfig(tol=config.tol, max_iter=config.max_iter)

    return ForbesQ2dGeometry(
        cs,
        surface_config=surface_config,
        solver_config=solver_config,
    )


def _create_nurbs(cs: CoordinateSystem, config: GeometryConfig):
    """Create a NURBS geometry."""

    return NurbsGeometry(
        cs,
        config.radius,
        config.conic,
        config.nurbs_norm_x,
        config.nurbs_norm_y,
        config.nurbs_x_center,
        config.nurbs_y_center,
        config.control_points,
        config.weights,
        config.u_degree,
        config.v_degree,
        config.u_knots,
        config.v_knots,
        config.n_points_u,
        config.n_points_v,
        config.tol,
        config.max_iter,
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
    "grating": _create_grating,
    "odd_asphere": _create_odd_asphere,
    "paraxial": _create_paraxial,
    "polynomial": _create_polynomial,
    "standard": _create_standard,
    "toroidal": _create_toroidal,
    "zernike": _create_zernike,
    "forbes_qbfs": _create_forbes_qbfs,
    "forbes_q2d": _create_forbes_q2d,
    "nurbs": _create_nurbs,
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
