import pytest
from PySide6.QtWidgets import QApplication
from optiland_gui.main_window import MainWindow

@pytest.fixture(scope="session")
def qapp():
    """Session-wide Qt Application."""
    return QApplication.instance() or QApplication([])

@pytest.fixture
def app(qtbot):
    """Create and tear down the main application window."""
    main_window = MainWindow()
    qtbot.addWidget(main_window)
    yield main_window
    main_window.close()
