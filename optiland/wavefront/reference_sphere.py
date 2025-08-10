"""
This module defines the reference sphere calculation strategies.

Kramer Harrison, 2024
"""

from abc import ABC, abstractmethod

import optiland.backend as be


class ReferenceSphereCalculator(ABC):
    """
    Abstract base class for reference sphere calculation strategies.
    """

    def __init__(self, optic):
        self.optic = optic

    @abstractmethod
    def calculate(self, pupil_z):
        """
        Calculate the reference sphere parameters.

        Args:
            pupil_z (float): The z-coordinate of the exit pupil.

        Returns:
            tuple: A tuple containing the reference sphere parameters
                   (xc, yc, zc, R).
        """
        pass


class ChiefRayReferenceSphereCalculator(ReferenceSphereCalculator):
    """
    Calculates the reference sphere based on the chief ray.
    """

    def calculate(self, pupil_z):
        """
        Determine reference sphere center and radius from chief ray.
        """
        x = self.optic.surface_group.x[-1, :]
        y = self.optic.surface_group.y[-1, :]
        z = self.optic.surface_group.z[-1, :]
        if be.size(x) != 1:
            raise ValueError("Chief ray cannot be determined. It must be traced alone.")
        R = be.sqrt(x**2 + y**2 + (z - pupil_z) ** 2)
        return x, y, z, R


def create_reference_sphere_calculator(strategy, optic):
    """
    Factory function to create a reference sphere calculator.

    Args:
        strategy (str): The name of the strategy to use.
        optic (Optic): The optical system.

    Returns:
        ReferenceSphereCalculator: An instance of the requested calculator.
    """
    if strategy == "chief_ray":
        return ChiefRayReferenceSphereCalculator(optic)
    else:
        raise ValueError(f"Unknown reference sphere strategy: {strategy}")
