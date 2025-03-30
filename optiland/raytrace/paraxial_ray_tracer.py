"""Paraxial Ray Tracer Module

This module contains the ParaxialRayTracer class, which is responsible for tracing
paraxial rays through an optical system.

Kramer Harrison, 2025
"""


class ParaxialRayTracer:
    """Class to trace paraxial rays through an optical system

    Args:
        optic (Optic): The optical system to be traced.
    """

    def __init__(self, optic):
        self.optic = optic
