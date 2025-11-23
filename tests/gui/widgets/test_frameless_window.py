import pytest
from PySide6.QtCore import QPoint
from optiland_gui.widgets.frameless_window import FramelessWindow

@pytest.fixture
def frameless_window(qtbot):
    """Fixture for a FramelessWindow instance."""
    window = FramelessWindow()
    window.resize(800, 600)
    qtbot.addWidget(window)
    return window

def test_get_resize_area(frameless_window):
    """
    Test that the _get_resize_area method returns the correct resize area.
    """
    grip_size = frameless_window.grip_size
    width = frameless_window.width()
    height = frameless_window.height()

    # Test the corners
    assert frameless_window._get_resize_area(QPoint(0, 0)) == "top_left"
    assert frameless_window._get_resize_area(QPoint(width - 1, 0)) == "top_right"
    assert frameless_window._get_resize_area(QPoint(0, height - 1)) == "bottom_left"
    assert frameless_window._get_resize_area(QPoint(width - 1, height - 1)) == "bottom_right"

    # Test the edges
    assert frameless_window._get_resize_area(QPoint(width / 2, 0)) == "top"
    assert frameless_window._get_resize_area(QPoint(0, height / 2)) == "left"
    assert frameless_window._get_resize_area(QPoint(width - 1, height / 2)) == "right"
    assert frameless_window._get_resize_area(QPoint(width / 2, height - 1)) == "bottom"

    # Test the center
    assert frameless_window._get_resize_area(QPoint(width / 2, height / 2)) is None
