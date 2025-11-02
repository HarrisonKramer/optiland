import pytest
from PySide6.QtWidgets import QWidget
from optiland_gui.viewer_panel import SagViewer
from unittest.mock import MagicMock

@pytest.fixture
def sag_viewer(qtbot, mocker):
    """Fixture for a SagViewer instance."""
    mocker.patch.object(SagViewer, 'plot_sag')
    connector = MagicMock()
    connector.get_surface_count.return_value = 1
    viewer = SagViewer(connector)
    qtbot.addWidget(viewer)
    return viewer

def test_sag_viewer_toggle_settings(sag_viewer, mocker):
    """
    Test that the _toggle_settings method calls setVisible on the settings widget.
    """
    spy = mocker.spy(sag_viewer.settings_area, 'setVisible')

    # Click to hide
    sag_viewer._toggle_settings(False)
    spy.assert_called_once_with(False)

    spy.reset_mock()

    # Click to show
    sag_viewer._toggle_settings(True)
    spy.assert_called_once_with(True)
