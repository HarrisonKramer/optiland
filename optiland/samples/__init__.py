"""This package provides a collection of predefined sample optical systems."""

from .simple import (
    Edmund_49_847,
    SingletStopSurf2,
    TelescopeDoublet,
    CementedAchromat,
    AsphericSinglet,
)
from .eyepieces import EyepieceErfle
from .infrared import (
    InfraredTriplet,
    InfraredTripletF4,
)
from .lithography import UVProjectionLens
from .microscopes import (
    Objective60x,
    Microscope20x,
    UVReflectingMicroscope,
)
from .objectives import (
    TripletTelescopeObjective,
    CookeTriplet,
    DoubleGauss,
    ReverseTelephoto,
    ObjectiveUS008879901,
    TelescopeObjective48Inch,
    HeliarLens,
    TessarLens,
    LensWithFieldCorrector,
    PetzvalLens,
    Telephoto,
)
from .telescopes import HubbleTelescope

__all__ = [
    # From simple.py
    "Edmund_49_847",
    "SingletStopSurf2",
    "TelescopeDoublet",
    "CementedAchromat",
    "AsphericSinglet",
    # From eyepieces.py
    "EyepieceErfle",
    # From infrared.py
    "InfraredTriplet",
    "InfraredTripletF4",
    # From lithography.py
    "UVProjectionLens",
    # From microscopes.py
    "Objective60x",
    "Microscope20x",
    "UVReflectingMicroscope",
    # From objectives.py
    "TripletTelescopeObjective",
    "CookeTriplet",
    "DoubleGauss",
    "ReverseTelephoto",
    "ObjectiveUS008879901",
    "TelescopeObjective48Inch",
    "HeliarLens",
    "TessarLens",
    "LensWithFieldCorrector",
    "PetzvalLens",
    "Telephoto",
    # From telescopes.py
    "HubbleTelescope",
]
