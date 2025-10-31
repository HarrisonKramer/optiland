from optiland_gui.main_window import MainWindow

def test_main_window_creation(app):
    """
    Test that the main window can be created without errors.
    """
    assert isinstance(app, MainWindow)
