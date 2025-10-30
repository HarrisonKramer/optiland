import pytest
from unittest.mock import MagicMock, call
from PySide6.QtWidgets import QMainWindow, QDockWidget, QWidget
from PySide6.QtCore import Qt

from optiland_gui.panel_manager import PanelManager
from optiland_gui.main_window import MainWindow

@pytest.fixture
def mocked_app(mocker, qtbot):
    """
    Creates a MainWindow instance with all panel classes mocked to return
    QWidget instances with the necessary mock attributes.
    """
    # Mock all panel classes to return a QWidget instance with mock attributes
    def mock_panel(*args, **kwargs):
        widget = QWidget()
        widget.menuSelected = MagicMock()
        widget.commandExecuted = MagicMock()
        widget.update_theme = MagicMock()
        widget.set_theme = MagicMock()
        widget.update_icons = MagicMock()
        widget.shutdown_kernel = MagicMock()
        return widget

    mocker.patch('optiland_gui.panel_manager.LensEditor', side_effect=mock_panel)
    mocker.patch('optiland_gui.panel_manager.SystemPropertiesPanel', side_effect=mock_panel)
    mocker.patch('optiland_gui.panel_manager.AnalysisPanel', side_effect=mock_panel)
    mocker.patch('optiland_gui.panel_manager.PythonTerminalWidget', side_effect=mock_panel)
    mocker.patch('optiland_gui.panel_manager.SidebarWidget', side_effect=mock_panel)

    # Now that panels are mocked, create the MainWindow instance
    main_window = MainWindow()
    qtbot.addWidget(main_window)
    yield main_window
    main_window.close()

@pytest.fixture
def panel_manager(mocked_app):
    """
    Returns the PanelManager from the mocked_app.
    """
    return mocked_app.panel_manager


def test_create_all_panels(panel_manager):
    """Test that PanelManager correctly instantiates all panels and docks."""
    assert isinstance(panel_manager.sidebar, QDockWidget)
    assert isinstance(panel_manager.viewer_dock, QDockWidget)
    assert isinstance(panel_manager.lens_editor_dock, QDockWidget)
    assert isinstance(panel_manager.system_properties_dock, QDockWidget)
    assert isinstance(panel_manager.analysis_dock, QDockWidget)
    assert isinstance(panel_manager.terminal_dock, QDockWidget)
    assert len(panel_manager.all_docks) == 6


def test_setup_default_layout(mocker, mocked_app, panel_manager):
    """Test that the default panel layout is applied correctly."""
    mocker.patch.object(mocked_app, 'addDockWidget')
    mocker.patch.object(mocked_app, 'splitDockWidget')
    mocker.patch.object(mocked_app, 'tabifyDockWidget')

    for dock in panel_manager.all_docks:
        mocker.patch.object(dock, 'setFloating')
        mocker.patch.object(dock, 'show')
        mocker.patch.object(dock, 'raise_')

    panel_manager.setup_default_layout()

    for dock in panel_manager.all_docks:
        dock.setFloating.assert_called_with(False)
        dock.show.assert_called()
        dock.raise_.assert_called()

    assert mocked_app.addDockWidget.called
    mocked_app.splitDockWidget.assert_has_calls([
        call(panel_manager.lens_editor_dock, panel_manager.analysis_dock, Qt.Horizontal),
        call(panel_manager.lens_editor_dock, panel_manager.viewer_dock, Qt.Vertical),
        call(panel_manager.analysis_dock, panel_manager.terminal_dock, Qt.Vertical),
    ], any_order=True)
    mocked_app.tabifyDockWidget.assert_called_with(
        panel_manager.analysis_dock, panel_manager.system_properties_dock
    )


def test_connect_signals(panel_manager):
    """Test that panel signals are correctly connected."""
    panel_manager.connect_signals()

    panel_manager.sidebar_content_widget.menuSelected.connect.assert_called_with(
        panel_manager.on_sidebar_menu_selected
    )
    panel_manager.python_terminal.commandExecuted.connect.assert_called_with(
        panel_manager.connector.opticChanged.emit
    )


@pytest.mark.parametrize("button_name, expected_dock_name", [
    ("analysis", "analysis_dock"),
    ("scripts", "terminal_dock"),
    ("design", "lens_editor_dock"),
])
def test_on_sidebar_menu_selected(panel_manager, button_name, expected_dock_name):
    """Test that the correct dock is shown when a sidebar button is clicked."""
    target_dock = getattr(panel_manager, expected_dock_name)
    target_dock.show = MagicMock()
    target_dock.raise_ = MagicMock()

    panel_manager.on_sidebar_menu_selected(button_name)

    target_dock.show.assert_called_once()
    target_dock.raise_.assert_called_once()


def test_update_theme(panel_manager):
    """Test that theme changes are propagated to all relevant panels."""
    theme = "light"
    panel_manager.update_theme(theme)

    panel_manager.sidebar_content_widget.update_icons.assert_called_with(theme)
    panel_manager.analysis_panel.update_theme.assert_called_with(theme)
    panel_manager.viewer_panel.update_theme.assert_called_with(theme)
    panel_manager.python_terminal.set_theme.assert_called_with(theme)
