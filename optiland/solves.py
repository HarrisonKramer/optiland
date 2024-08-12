from abc import ABC, abstractmethod


class BaseSolve(ABC):
    """
    Applies a solve operation.

    This method should be implemented by subclasses to define the specific
    behavior of the solve operation.

    Raises:
        NotImplementedError: If the method is not implemented by the subclass.
    """
    @abstractmethod
    def apply(self):
        pass


class MarginalRayHeightSolve(BaseSolve):
    """
    Initializes a MarginalRayHeightSolve object.

    Args:
        optic (Optic): The optic object.
        surface_idx (int): The index of the surface.
        height (float): The height of the ray.
    """
    def __init__(self, optic, surface_idx, height):
        self.optic = optic
        self.surface_idx = surface_idx
        self.height = height

    def apply(self):
        """Applies the MarginalRayHeightSolve to the optic."""
        ya, ua = self.paraxial.marginal_ray()
        offset = ya[self.surface_idx] / ua[self.surface_idx]

        # shift current surface and all subsequent surfaces
        for surface in self.optic.surface_group.surfaces[self.surface_idx:]:
            surface.geometry.cs.z -= offset
