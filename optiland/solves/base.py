"""Base Solve Module

Defines the base class for solve operations.

Kramer Harrison, 2025
"""

from abc import ABC, abstractmethod


class BaseSolve(ABC):
    """Applies a solve operation.

    This method should be implemented by subclasses to define the specific
    behavior of the solve operation.

    Raises:
        NotImplementedError: If the method is not implemented by the subclass.

    """

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseSolve._registry[cls.__name__] = cls

    @abstractmethod
    def apply(self):
        """Applies the solve operation to the optic."""
        pass  # pragma: no cover

    def to_dict(self):
        """Returns a dictionary representation of the solve.

        Returns:
            dict: A dictionary representation of the solve.

        """
        return {
            "type": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, optic, data):
        """Creates a solve from a dictionary representation.

        Args:
            optic (Optic): The optic object.
            data (dict): The dictionary representation of the solve.

        Returns:
            BaseSolve: The solve.

        """
        solve_type = data["type"]
        if solve_type not in BaseSolve._registry:
            raise ValueError(f"Unknown solve type: {solve_type}")
        solve_class = BaseSolve._registry[data["type"]]
        return solve_class.from_dict(optic, data)
