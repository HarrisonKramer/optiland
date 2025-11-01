import pytest
from PySide6.QtWidgets import QApplication, QWidget
from optiland_gui.main_window import MainWindow

@pytest.fixture(scope="session")
def qapp():
    """Session-wide Qt Application."""
    return QApplication.instance() or QApplication([])

@pytest.fixture(autouse=True)
def mock_viewer_panel(mocker):
    """
    Mock the ViewerPanel to prevent VTK from being initialized during tests,
    which can cause crashes in headless environments.
    The mock needs to return a QWidget instance to be valid for setWidget.
    It also needs to have the attributes that MainWindow expects.
    """
    mock_class = mocker.patch('optiland_gui.panel_manager.ViewerPanel', autospec=True)

    # Create a real QWidget to satisfy type checks
    viewer_panel_instance = QWidget()

    # Add mock attributes that MainWindow and PanelManager access
    viewer_panel_instance.viewer2D = mocker.MagicMock()
    viewer_panel_instance.viewer3D = mocker.MagicMock()
    viewer_panel_instance.update_theme = mocker.MagicMock()

    # Configure the mock class to return our instance
    mock_class.return_value = viewer_panel_instance


@pytest.fixture
def app(qtbot):
    """Create and tear down the main application window."""
    main_window = MainWindow()
    qtbot.addWidget(main_window)
    yield main_window
    main_window.close()
