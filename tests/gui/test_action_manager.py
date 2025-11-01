import pytest
from PySide6.QtGui import QKeySequence, QActionGroup

from optiland_gui.action_manager import ActionManager
from optiland_gui.config import THEME_DARK_PATH, THEME_LIGHT_PATH


@pytest.fixture
def manager(mocker, app):
    """
    Returns an ActionManager instance with mocked MainWindow methods.
    This fixture patches the methods on the 'app' instance and returns
    a manager instance WITHOUT creating the actions. This allows tests
    to set up preconditions before actions are created.
    """
    mocker.patch.object(app, 'new_system_action')
    mocker.patch.object(app, 'open_system_action')
    mocker.patch.object(app, 'save_system_action')
    mocker.patch.object(app, 'save_system_as_action')
    mocker.patch.object(app, 'close')
    mocker.patch.object(app.connector, 'undo')
    mocker.patch.object(app.connector, 'redo')
    mocker.patch.object(app, 'reset_windows_action')
    mocker.patch.object(app, 'load_layout_1_slot')
    mocker.patch.object(app, 'load_layout_2_slot')
    mocker.patch.object(app, 'save_layout_slot')
    mocker.patch.object(app, 'switch_theme')
    mocker.patch.object(app, 'about_action')
    # Mock the 'contains' method of the real QSettings object
    mocker.patch.object(app.settings, 'contains')

    action_manager = ActionManager(app, app.connector)
    return action_manager


def test_create_file_actions(app, manager):
    """Test that file-related actions are created correctly."""
    manager.create_all_actions()

    manager.get_action("new").trigger()
    app.new_system_action.assert_called_once()

    manager.get_action("open").trigger()
    app.open_system_action.assert_called_once()

    manager.get_action("save").trigger()
    app.save_system_action.assert_called_once()

    manager.get_action("save_as").trigger()
    app.save_system_as_action.assert_called_once()

    manager.get_action("exit").trigger()
    app.close.assert_called_once()


def test_create_edit_actions(app, manager):
    """Test that edit-related actions are created and connected correctly."""
    manager.create_all_actions()

    undo_action = manager.get_action("undo")
    assert not undo_action.isEnabled()
    undo_action.setEnabled(True)  # Enable manually for test
    undo_action.trigger()
    app.connector.undo.assert_called_once()

    redo_action = manager.get_action("redo")
    assert not redo_action.isEnabled()
    redo_action.setEnabled(True)  # Enable manually for test
    redo_action.trigger()
    app.connector.redo.assert_called_once()


def test_create_view_actions(app, manager):
    """Test that view-related actions are created correctly."""
    manager.create_all_actions()

    manager.get_action("dock_all").trigger()
    app.reset_windows_action.assert_called()

    manager.get_action("reset_layout").trigger()
    app.reset_windows_action.assert_called()


def test_create_layout_actions_enabled(app, manager):
    """Test layout actions when settings exist."""
    app.settings.contains.return_value = True
    manager.create_all_actions()

    load1_action = manager.get_action("load_layout_1")
    assert load1_action.isEnabled()
    load1_action.trigger()
    app.load_layout_1_slot.assert_called_once()

    load2_action = manager.get_action("load_layout_2")
    assert load2_action.isEnabled()
    load2_action.trigger()
    app.load_layout_2_slot.assert_called_once()

    manager.get_action("save_layout").trigger()
    app.save_layout_slot.assert_called_once()


def test_create_layout_actions_disabled(app, manager):
    """Test layout actions when settings do not exist."""
    app.settings.contains.return_value = False
    manager.create_all_actions()

    assert not manager.get_action("load_layout_1").isEnabled()
    assert not manager.get_action("load_layout_2").isEnabled()


def test_create_theme_actions(app, manager):
    """Test that theme-related actions are created correctly."""
    manager.create_all_actions()

    manager.get_action("dark_theme").trigger()
    app.switch_theme.assert_called_with(THEME_DARK_PATH)

    manager.get_action("light_theme").trigger()
    app.switch_theme.assert_called_with(THEME_LIGHT_PATH)


def test_create_help_actions(app, manager):
    """Test that help-related actions are created correctly."""
    manager.create_all_actions()

    manager.get_action("about").trigger()
    app.about_action.assert_called_once()


def test_get_actions(manager):
    """Test retrieving multiple actions."""
    manager.create_all_actions()

    actions = manager.get_actions("new", "open", "exit")
    assert len(actions) == 3
    assert actions[0].text() == "&New System"
    assert actions[1].text() == "&Open System..."
    assert actions[2].text() == "E&xit"
