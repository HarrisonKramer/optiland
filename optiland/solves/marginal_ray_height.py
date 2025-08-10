"""Marginal Ray Height Solve Module

Defines the marginal ray height solve.

Kramer Harrison, 2025
"""

from optiland.solves.ray_height_base import RayHeightSolveBase


class MarginalRayHeightSolve(RayHeightSolveBase):
    """Solves for a target marginal ray height on a specific surface.

    This class adjusts the z-position of the specified surface and all
    subsequent surfaces to achieve the desired marginal ray height.
    """

    def __init__(self, optic, surface_idx: int, height: float):
        """Initializes a MarginalRayHeightSolve object.

        Args:
            optic (Optic): The optic object.
            surface_idx (int): The index of the surface where the marginal
                ray height is to be controlled.
            height (float): The target height of the marginal ray on the
                specified surface.
        """
        super().__init__(optic, surface_idx, height)

    def _get_ray_y_u(self):
        """Gets the marginal ray height (y) and slope (u) arrays.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - ya: marginal ray heights at each surface.
                - ua: marginal ray slopes before each surface.
        """
        return self.optic.paraxial.marginal_ray()
