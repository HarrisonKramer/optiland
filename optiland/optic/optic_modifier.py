"""Optic Modifier Module

This module contains the OpticModifier class, which is responsible for modifying the
optical system properties, such as the surface radii of curvature, thicknesses,
materials, conic constants, polarization, etc.

Kramer Harrison, 2025
"""


class OpticModifier:
    """Class to modify an optical system

    This class is responsible for modifying the optical system properties, such as
    the surface radii of curvature, thicknesses, materials, conic constants,
    polarization, etc.

    Args:
        optic (Optic): The optical system to be modified.
    """

    def __init__(self, optic):
        self.optic = optic
