"""Defines the chief ray height solve.

This module provides the `ChiefRayHeightSolve` class, which is used to
adjust an optical system to achieve a target chief ray height on a
specific surface.
"""

from optiland.solves.ray_height_base import _RayHeightSolveBase
# No need for 'import optiland.backend as be' here if not used directly.


class ChiefRayHeightSolve(_RayHeightSolveBase):
    """Solves for a target chief ray height on a specific surface.

    This class adjusts the z-position of the specified surface and all
    subsequent surfaces to achieve the desired chief ray height (yc). It
    operates similarly to `MarginalRayHeightSolve` but uses the chief ray
    parameters.
    """

    def __init__(self, optic, surface_idx: int, height: float):
        """Initializes a ChiefRayHeightSolve object.

        Args:
            optic (Optic): The optic object.
            surface_idx (int): The index of the surface where the chief
                ray height is to be controlled.
            height (float): The target height of the chief ray (yc) on the
                specified surface.
        """
        super().__init__(optic, surface_idx, height)

    def _get_ray_y_u(self):
        """Gets the chief ray height (yc) and slope (uc) arrays.

        Although the base class method is named `_get_ray_y_u`, for chief rays,
        these correspond to yc (chief ray height) and uc (chief ray slope).

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - yc: chief ray heights at each surface.
                - uc: chief ray slopes before each surface.
        """
        return self.optic.paraxial.chief_ray()
