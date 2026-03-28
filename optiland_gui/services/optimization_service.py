"""Optimization service stub for the Optiland GUI.

This module provides a stub implementation of ``OptimizationService``. Full
variable/operand management and threaded optimizer execution will be
implemented in Phase 6.
"""

from __future__ import annotations


class OptimizationService:
    """Manages optimization variables, operands, and optimizer execution.

    This is a stub implementation. Full functionality (variable/operand CRUD,
    ``OptimizationProblem`` construction, ``QThread``-based execution) will be
    added in Phase 6.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`
            instance that owns this service.
    """

    def __init__(self, connector: object) -> None:
        self._connector = connector

    def add_variable(self, variable: object) -> None:
        """Register an optimization variable.

        Args:
            variable: The variable descriptor to add.
        """

    def remove_variable(self, index: int) -> None:
        """Remove an optimization variable by index.

        Args:
            index: Zero-based index of the variable to remove.
        """

    def add_operand(self, operand: object) -> None:
        """Register an optimization operand.

        Args:
            operand: The operand descriptor to add.
        """

    def remove_operand(self, index: int) -> None:
        """Remove an optimization operand by index.

        Args:
            index: Zero-based index of the operand to remove.
        """

    def build_problem(self, optic: object) -> object:
        """Construct an ``OptimizationProblem`` from the registered variables
        and operands.

        Args:
            optic: The :class:`~optiland.optic.Optic` instance to optimise.

        Returns:
            ``None`` in the stub implementation.
        """
        return None

    def run(self) -> None:
        """Start the optimizer in a background thread."""

    def stop(self) -> None:
        """Request cancellation of an in-progress optimization run."""
