"""Chief Ray Height Solve Module

This module provides the `ChiefRayHeightSolve` class, which is used to
adjust an optical system to achieve a target chief ray height on a
specific surface.

Kramer Harrison, 2025
"""

from optiland.solves.ray_height_base import RayHeightSolveBase


class ChiefRayHeightSolve(RayHeightSolveBase):
    """Solves for a target chief ray height on a specific surface.

    This class adjusts the z-position of the specified surface and all
    subsequent surfaces to achieve the desired chief ray height (yb). It
    operates similarly to `MarginalRayHeightSolve` but uses the chief ray
    parameters.
    """

    def __init__(self, optic, surface_idx: int, height: float):
        """Initializes a ChiefRayHeightSolve object.

        Args:
            optic (Optic): The optic object.
            surface_idx (int): The index of the surface where the chief
                ray height is to be controlled.
            height (float): The target height of the chief ray (yb) on the
                specified surface.
        """
        super().__init__(optic, surface_idx, height)

    def _get_ray_y_u(self):
        """Gets the chief ray height (yb) and slope (ub) arrays.

        Although the base class method is named `_get_ray_y_u`, for chief rays,
        these correspond to yb (chief ray height) and ub (chief ray slope).

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - yb: chief ray heights at each surface.
                - ub: chief ray slopes before each surface.
        """
        return self.optic.paraxial.chief_ray()
