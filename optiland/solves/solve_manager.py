"""Solve Manager Module

Manages the application of solves to an optic.

Kramer Harrison, 2025
"""

from optiland.solves.base import BaseSolve
from optiland.solves.factory import SolveFactory


class SolveManager:
    """Manages the application of solves to an optic.

    Args:
        optic (Optic): The optic object

    Attributes:
        solves (list): A list of solve instances.

    Methods:
        add(solve_type, surface_idx, *args, **kwargs): Adds a solve
            instance to the list of solves.
        apply(): Applies all solves in the list.

    """

    def __init__(self, optic):
        self.optic = optic
        self.solves = []

    def __len__(self):
        return len(self.solves)

    def add(self, solve_type, surface_idx=None, *args, **kwargs):
        """Adds a solve instance to the list of solves.

        Args:
            solve_type (str): The type of solve to create.
            surface_idx (int): The index of the surface.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        solve = SolveFactory.create_solve(
            self.optic,
            solve_type,
            surface_idx,
            *args,
            **kwargs,
        )
        solve.apply()
        self.solves.append(solve)

    def apply(self):
        """Applies all solves in the list."""
        for solve in self.solves:
            solve.apply()

    def clear(self):
        """Clears the list of solves."""
        self.solves.clear()

    def remap_surface_indices(self, remap_func):
        """Remaps the surface indices of all relevant solves.

        Args:
            remap_func (callable): A function that takes an old surface index
                                   and returns a new surface index.
        """
        for solve in self.solves:
            if hasattr(solve, "surface_idx"):
                solve.surface_idx = remap_func(solve.surface_idx)

    def to_dict(self):
        """Returns a dictionary representation of the solve manager.

        Returns:
            dict: A dictionary representation of the solve manager.

        """
        return {"solves": [solve.to_dict() for solve in self.solves]}

    @classmethod
    def from_dict(cls, optic, data):
        """Creates a SolveManager from a dictionary representation.

        Args:
            optic (Optic): The optic object.
            data (dict): The dictionary representation of the solve manager.

        Returns:
            SolveManager: The solve manager.

        """
        solve_manager = cls(optic)
        for solve_data in data["solves"]:
            solve = BaseSolve.from_dict(optic, solve_data)
            solve_manager.solves.append(solve)
        return solve_manager
