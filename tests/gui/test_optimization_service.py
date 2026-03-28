"""Tests for OptimizationService."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from optiland_gui.services.optimization_service import OptimizationService


@pytest.fixture()
def mock_connector(minimal_optic):
    conn = MagicMock()
    conn._optic = minimal_optic
    conn._capture_optic_state.return_value = {}
    conn._undo_redo_manager = MagicMock()
    return conn


@pytest.fixture()
def service(mock_connector):
    return OptimizationService(mock_connector)


class TestOptimizationService:
    def test_add_variable(self, service):
        service.add_variable({"surface_number": 1, "type": "radius"})
        assert len(service.get_variables()) == 1
        assert service.get_variables()[0]["type"] == "radius"

    def test_remove_variable(self, service):
        service.add_variable({"surface_number": 1, "type": "radius"})
        service.add_variable({"surface_number": 2, "type": "thickness"})
        service.remove_variable(0)
        variables = service.get_variables()
        assert len(variables) == 1
        assert variables[0]["type"] == "thickness"

    def test_add_operand(self, service):
        service.add_operand(
            {
                "type": "total_track",
                "category": "Paraxial",
                "target": 50.0,
                "weight": 1.0,
                "input_data_str": "{}",
            }
        )
        assert len(service.get_operands()) == 1
        assert service.get_operands()[0]["type"] == "total_track"

    def test_build_problem_counts(self, service, minimal_optic):
        service.add_variable({"surface_number": 1, "type": "radius"})
        service.add_operand(
            {
                "type": "total_track",
                "category": "Paraxial",
                "target": 50.0,
                "weight": 1.0,
                "input_data_str": "{}",
            }
        )
        problem = service.build_problem(minimal_optic)
        assert len(problem.variables) == 1
        assert len(problem.operands) == 1

    def test_least_squares_run_does_not_raise(self, service, minimal_optic, qapp):
        """Synchronous LeastSquares run should complete without error."""
        from optiland.optimization.optimizer.scipy import LeastSquares

        service.add_variable({"surface_number": 1, "type": "radius"})
        service.add_operand(
            {
                "type": "total_track",
                "category": "Paraxial",
                "target": 50.0,
                "weight": 1.0,
                "input_data_str": "{}",
            }
        )
        problem = service.build_problem(minimal_optic)
        optimizer = LeastSquares(problem)
        # Run synchronously with a very low iteration limit
        optimizer.optimize(maxiter=2)
