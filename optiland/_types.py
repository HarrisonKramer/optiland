from typing import Literal, TypedDict

from numpy.typing import NDArray

from optiland.coatings import BaseCoating
from optiland.physical_apertures.base import BaseAperture

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