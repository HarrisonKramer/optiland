import pytest
from PySide6.QtWidgets import QWidget
@pytest.fixture
def optimization_panel(app):
    """Returns the OptimizationPanel instance from the main application window."""
    return app.panel_manager.optimization_panel

def test_optimization_panel_creation(optimization_panel):
    """
    Test that the OptimizationPanel widget can be created without errors.
    """
    assert isinstance(optimization_panel, QWidget)

def test_start_optimization_button(mocker, optimization_panel):
    """
    Test that clicking the 'Start Optimization' button calls the start_optimization method.
    """
    mocker.patch.object(optimization_panel, 'start_optimization')

    # Simulate a user clicking the run button
    optimization_panel.btnStartOptimization.click()

    optimization_panel.start_optimization.assert_called_once()
