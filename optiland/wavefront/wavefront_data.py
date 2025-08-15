"""
This module defines the WavefrontData class.

Kramer Harrison, 2024
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optiland.backend as be


@dataclass
class WavefrontData:
    """
    Data container for wavefront results at a given field and wavelength.

    Attributes:
        pupil_x (be.ndarray): x-coordinates of ray intersections at exit pupil.
        pupil_y (be.ndarray): y-coordinates of ray intersections at exit pupil.
        pupil_z (be.ndarray): z-coordinates of ray intersections at exit pupil.
        opd (be.ndarray): Optical path difference data, normalized to waves.
        intensity (be.ndarray): Ray intensities at the exit pupil.
        radius (be.ndarray): Radius of curvature of the exit pupil reference sphere.
    """

    pupil_x: be.ndarray
    pupil_y: be.ndarray
    pupil_z: be.ndarray
    opd: be.ndarray
    intensity: be.ndarray
    radius: be.ndarray
