import pytest
from PySide6.QtWidgets import QWidget

@pytest.fixture
def sidebar(app):
    """Returns the SidebarWidget instance from the main application window."""
    return app.panel_manager.sidebar_content_widget

def test_sidebar_widget_creation(sidebar):
    """
    Test that the SidebarWidget can be created without errors.
    """
    assert isinstance(sidebar, QWidget)

@pytest.mark.parametrize("button_name", ["scripts", "design"])
def test_sidebar_button_clicks(qtbot, sidebar, button_name):
    """
    Test that clicking each sidebar button emits the menuSelected signal
    with the correct button name.
    """
    button_item = next((b for b in sidebar._buttons_list if b["name"] == button_name), None)
    assert button_item is not None
    button = button_item["widget"]

    with qtbot.wait_signal(sidebar.menuSelected, check_params_cb=lambda name: name == button_name):
        button.click()

@pytest.mark.parametrize("button_name", ["analysis"])
def test_wip_sidebar_button_clicks(qtbot, mocker, sidebar, button_name):
    """
    Test that clicking a work-in-progress button does not emit the
    menuSelected signal.
    """
    mocker.patch('PySide6.QtWidgets.QMessageBox.information')
    button_item = next((b for b in sidebar._buttons_list if b["name"] == button_name), None)
    assert button_item is not None
    button = button_item["widget"]

    with qtbot.assertNotEmitted(sidebar.menuSelected):
        button.click()

def test_theme_update(sidebar):
    """
    Test that the update_icons method correctly updates the button icons.
    """
    # Set the theme to "light" and verify the icons are not null
    sidebar.update_icons("light")
    for item in sidebar._buttons_list:
        icon = item["widget"].icon()
        assert not icon.isNull()

    # Set the theme back to "dark" and verify the icons are not null
    sidebar.update_icons("dark")
    for item in sidebar._buttons_list:
        icon = item["widget"].icon()
        assert not icon.isNull()
