"""Factory for creating solve objects."""

from optiland.solves.marginal_ray_height import MarginalRayHeightSolve
from optiland.solves.quick_focus import QuickFocusSolve
from optiland.solves.chief_ray_height import ChiefRayHeightSolve
# BaseSolve is no longer needed here as we are moving away from the registry.
# The individual solve modules (marginal_ray_height, quick_focus, chief_ray_height)
# are still imported so their classes are defined for the _solve_map.
# However, the __init_subclass__ registration in BaseSolve is now bypassed by this factory.


class SolveFactory:
    """Factory class for creating solves.

    This factory uses a predefined map to find and instantiate
    the appropriate solve class based on the provided `solve_type`.
    """

    _solve_map = {
        "marginal_ray_height": MarginalRayHeightSolve,
        "quick_focus": QuickFocusSolve,
        "chief_ray_height": ChiefRayHeightSolve,
    }

    @staticmethod
    def create_solve(optic, solve_type: str, surface_idx, *args, **kwargs):
        """Creates a solve instance based on the given solve type.

        Args:
            optic (Optic): The optic object to which the solve will be applied.
            solve_type (str): The type of solve to create (e.g.,
                'marginal_ray_height', 'quick_focus', 'chief_ray_height').
            surface_idx (int): The index of the surface relevant to the solve.
                May not be used by all solve types (e.g., 'quick_focus').
            *args: Variable length argument list. For height-based solves,
                   the first argument (args[0]) is expected to be the height.
            **kwargs: Arbitrary keyword arguments. (Currently unused in favor
                      of positional arguments for height).

        Returns:
            An instance of the solve class corresponding to the given solve type.

        Raises:
            ValueError: If the solve type is invalid or if required arguments
                for a specific solve type are missing or invalid.
        """
        solve_class = SolveFactory._solve_map.get(solve_type)

        if solve_class is None:
            raise ValueError(f"Unknown solve type: {solve_type}")

        if solve_class is MarginalRayHeightSolve or solve_class is ChiefRayHeightSolve:
            if not args:
                raise ValueError(
                    f"Missing 'height' argument (expected as first positional "
                    f"argument) for {solve_type} solve."
                )
            # TODO: Consider adding type checking for args[0] if it should be float/int
            height = args[0]
            return solve_class(optic, surface_idx, height)
        elif solve_class is QuickFocusSolve:
            # QuickFocusSolve constructor takes only 'optic'.
            # surface_idx, args, and kwargs are ignored for this type.
            return solve_class(optic)
        else:
            # This case should ideally not be reached if _solve_map is comprehensive
            # and the initial check for solve_class is None handles unknown types.
            # However, as a safeguard:
            raise ValueError(
                f"Unhandled solve type in factory logic: {solve_type}"
            )
