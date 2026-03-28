"""Analysis runner service stub for the Optiland GUI.

This module provides a stub implementation of ``AnalysisRunner``. Full
analysis discovery and execution will be implemented in Phase 2/3.
"""

from __future__ import annotations


class AnalysisRunner:
    """Manages analysis discovery, parameter binding, and execution lifecycle.

    This is a stub implementation. Full functionality (registry loading via
    ``importlib``, threaded execution, result caching) will be added in a
    later phase.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`
            instance that owns this service.
    """

    def __init__(self, connector: object) -> None:
        self._connector = connector

    def get_analysis_registry(self) -> list:
        """Return the list of registered analyses.

        Returns:
            An empty list in the stub implementation.
        """
        return []

    def run(
        self,
        analysis_name: str,
        params: dict,
        optic: object,
    ) -> None:
        """Execute a named analysis with the given parameters.

        Args:
            analysis_name: The display name or class path of the analysis.
            params: A dict of parameter name → value pairs to pass to the
                analysis class constructor.
            optic: The :class:`~optiland.optic.Optic` instance to analyse.
        """

    def stop(self) -> None:
        """Request cancellation of an in-progress analysis run."""

    def get_result(self) -> object:
        """Return the result of the most recent analysis run.

        Returns:
            ``None`` in the stub implementation.
        """
        return None
