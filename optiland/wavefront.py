"""Wavefront Module

This module is deprecated and will be removed in a future version.
Please use `from optiland.wavefront import ...` instead.

Kramer Harrison, 2024
"""

import warnings

from optiland.wavefront.opd import OPD
from optiland.wavefront.opd_fan import OPDFan
from optiland.wavefront.wavefront import Wavefront
from optiland.wavefront.wavefront_data import WavefrontData
from optiland.wavefront.zernike_opd import ZernikeOPD

warnings.warn(
    "The 'optiland.wavefront' module is deprecated and will be removed in a "
    "future version. Please use 'from optiland.wavefront import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Wavefront",
    "WavefrontData",
    "OPD",
    "OPDFan",
    "ZernikeOPD",
]
