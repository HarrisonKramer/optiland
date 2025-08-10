"""Ray Height Base Module

This module defines `RayHeightSolveBase`, an abstract base class for solves
that adjust surface positions to achieve a target ray height on a specific
surface.

Kramer Harrison, 2025
"""

from abc import ABC, abstractmethod

import optiland.backend as be
from optiland.solves.base import BaseSolve


class RayHeightSolveBase(BaseSolve, ABC):
    """Abstract base class for ray height solves.

    This class provides the common structure for solves that aim to achieve a
    specific ray height at a given surface by adjusting the z-position of that
    surface and subsequent surfaces.

    Attributes:
        optic (Optic): The optic object.
        surface_idx (int): The index of the surface where the ray height
            is to be controlled.
        height (float): The target height of the ray on the specified surface.
    """

    def __init__(self, optic, surface_idx: int, height: float):
        """Initializes a _RayHeightSolveBase object.

        Args:
            optic (Optic): The optic object.
            surface_idx (int): The index of the surface.
            height (float): The target height of the ray.
        """
        if surface_idx is None:
            raise ValueError("'surface_idx' argument must be provided.")
        super().__init__()
        self.optic = optic
        self.surface_idx = surface_idx
        self.height = height

    @abstractmethod
    def _get_ray_y_u(self):
        """Gets the ray height (y) and slope (u) arrays for the relevant ray.

        This method must be implemented by subclasses to provide the specific
        ray data (e.g., marginal ray, chief ray) used by the solve.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - ya: ray heights at each surface.
                - ua: ray slopes before each surface.
        """
        pass  # pragma: no cover

    def apply(self):
        """Applies the ray height solve to the optic.

        This method calculates the necessary shift in z-position for the
        target surface and all subsequent surfaces to achieve the desired
        ray height.
        """
        y, u = self._get_ray_y_u()

        # Ensure surface_idx is within bounds for y and u
        if not (0 <= self.surface_idx < len(y) and 0 <= self.surface_idx < len(u)):
            raise IndexError(
                f"surface_idx {self.surface_idx} is out of bounds for ray data arrays "
                f"of length {len(y)}."
            )

        if u[self.surface_idx] == 0:
            # Handle case where ray is parallel to optical axis at the surface
            # Or provide a more specific error/warning
            # For now, let's assume this implies an issue or no change possible
            # Or raise an error if this state is unexpected for a solve
            be.logging.warning(
                f"Ray slope is zero at surface {self.surface_idx}; "
                "cannot apply height solve."
            )
            return

        offset = (self.height - y[self.surface_idx]) / u[self.surface_idx]

        # Shift current surface and all subsequent surfaces
        num_surfaces_in_group = len(self.optic.surface_group.surfaces)
        if not (0 <= self.surface_idx < num_surfaces_in_group):
            raise IndexError(
                f"surface_idx {self.surface_idx} is out of bounds for surface group "
                f"of length {num_surfaces_in_group}."
            )

        for i in range(self.surface_idx, num_surfaces_in_group):
            surface = self.optic.surface_group.surfaces[i]
            # Perform read, compute, and write separately to avoid potential += issues
            current_z = surface.geometry.cs.z
            new_z = current_z + offset  # offset should be a scalar here
            surface.geometry.cs.z = new_z

    def to_dict(self):
        """Returns a dictionary representation of the solve.

        Returns:
            dict: A dictionary representation of the solve, including
                  surface index and target height.
        """
        solve_dict = super().to_dict()
        solve_dict.update(
            {
                "surface_idx": self.surface_idx,
                "height": self.height,
            }
        )
        return solve_dict

    @classmethod
    def from_dict(cls, optic, data):
        """Creates a solve instance from a dictionary representation.

        This method is typically called by `BaseSolve.from_dict` when
        reconstructing registered subclasses.

        Args:
            optic (Optic): The optic object.
            data (dict): The dictionary representation of the solve,
                         expected to contain 'surface_idx' and 'height'.

        Returns:
            _RayHeightSolveBase: An instance of the specific ray height solve subclass.
        """
        if cls is RayHeightSolveBase:
            raise TypeError(
                "RayHeightSolveBase is an abstract class and cannot be "
                "instantiated directly."
            )
        return cls(optic, data["surface_idx"], data["height"])
