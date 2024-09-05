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
        pass  # pragma: no cover


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
        ya, ua = self.optic.paraxial.marginal_ray()
        offset = (self.height - ya[self.surface_idx]) / ua[self.surface_idx]

        # shift current surface and all subsequent surfaces
        for surface in self.optic.surface_group.surfaces[self.surface_idx:]:
            surface.geometry.cs.z += offset


class SolveFactory:
    """
    Factory class for creating solves.

    Attributes:
        _solve_map (dict): A dictionary mapping solve types to solve classes.

    Methods:
        create_solve(solve_type, *args, **kwargs): Creates a solve instance
            based on the given solve type.
    """
    _solve_map = {
        'marginal_ray_height': MarginalRayHeightSolve
    }

    @staticmethod
    def create_solve(optic, solve_type, surface_idx, *args, **kwargs):
        """
        Creates a solve instance based on the given solve type.

        Args:
            optic (Optic): The optic object.
            solve_type (str): The type of solve to create.
            surface_idx (int): The index of the surface.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            solve_instance: An instance of the solve class corresponding to
                the given solve type.

        Raises:
            ValueError: If the solve type is invalid.
        """
        solve_class = SolveFactory._solve_map.get(solve_type)
        if solve_class is None:
            raise ValueError(f'Invalid solve type: {solve_type}')
        return solve_class(optic, surface_idx, *args, **kwargs)
