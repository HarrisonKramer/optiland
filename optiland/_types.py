from __future__ import annotations

from sys import version_info
from typing import TYPE_CHECKING, Literal, TypedDict

from numpy.typing import NDArray

if TYPE_CHECKING:
    from optiland.coatings import BaseCoating
    from optiland.physical_apertures.base import BaseAperture

if version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack

__all__ = [
    "DistributionType",
    "ApertureType",
    "FieldType",
    "ReferenceRay",
    "WavelengthUnit",
    "FloatOrArray",
    "SurfaceType",
    "SurfaceParameters",
    "Unpack",
    "ZernikeType",
]

DistributionType = Literal[
    "line_x",
    "line_y",
    "positive_line_x",
    "positive_line_y",
    "random",
    "uniform",
    "hexapolar",
    "cross",
    "ring",
]
ApertureType = Literal["EPD", "imageFNO", "objectNA", "float_by_stop_size"]
FieldType = Literal["angle", "object_height"]
ReferenceRay = Literal["chief", "marginal"]
WavelengthUnit = Literal["nm", "um", "mm", "cm", "m"]
ZernikeType = Literal["standard", "noll", "fringe"]

FloatOrArray = float | NDArray

SurfaceType = Literal[
    "biconic",
    "chebyshev",
    "even_asphere",
    "odd_asphere",
    "paraxial",
    "polynomial",
    "standard",
    "toroidal",
    "zernike",
    "grating",
]


class SurfaceParameters(TypedDict, total=False):
    # Geometry parameters
    radius: float
    conic: float
    coefficients: list[float]
    tol: float
    max_iter: int
    norm_x: float
    norm_y: float
    norm_radius: float
    radius_x: float
    radius_y: float
    conic_x: float
    conic_y: float
    toroidal_coeffs_poly_y: list[float]
    zernike_type: ZernikeType
    radial_terms: dict[int, float]
    freeform_coeffs: dict[tuple[int, int] | tuple[int, int, Literal["sin"]], float]
    forbes_norm_radius: float
    grating_order: int
    grating_period: float
    groove_orientation_angle: float

    # Coordinate system parameters
    thickness: float
    x: float
    y: float
    z: float
    dx: float
    dy: float
    rx: float
    ry: float
    rz: float

    # Coating parameters
    coating: str | BaseCoating

    # Paraxial parameters
    f: float  # focal length
    aperture: BaseAperture
