
def test_new_system_action(app, qtbot):
    """
    Test that triggering the 'New System' action resets the application state.
    """
    # Pre-condition: Add some data to the lens editor to ensure it gets cleared
    lens_editor = app.panel_manager.lens_editor
    lens_editor.tableWidget.insertRow(0)
    assert lens_editor.tableWidget.rowCount() > 0

    # Trigger the action
    new_action = app.action_manager.get_action("new")
    new_action.trigger()

    # Verify the state is reset
    assert lens_editor.tableWidget.rowCount() == 3 # A new system has 3 surfaces
    assert "New Untitled System" in app.windowTitle()

def test_add_surface_action(app, qtbot):
    """
    Test that triggering the 'Add Surface' action adds a new surface to the
    optic model and updates the lens editor.
    """
    initial_surface_count = app.connector.get_surface_count()

    # Trigger the action
    add_action = app.panel_manager.lens_editor.btnAddSurface
    add_action.click()

    # Verify the optic model and lens editor are updated
    assert app.connector.get_surface_count() == initial_surface_count + 1
    assert app.panel_manager.lens_editor.tableWidget.rowCount() == initial_surface_count + 1

def test_update_project_name_in_title_bar(app, mocker):
    """
    Test that the project name is updated in the title bar.
    """
    mocker.patch.object(app.custom_title_bar_widget, 'set_project_name')
    # Test with a new system
    app.connector.new_system()
    app._update_project_name_in_title_bar()
    app.custom_title_bar_widget.set_project_name.assert_called_with("New Untitled System")
    # Test with a loaded file
    app.connector._current_filepath = "test.json"
    app._update_project_name_in_title_bar()
    app.custom_title_bar_widget.set_project_name.assert_called_with("test.json")
    # Test with a modified file
    app.connector.set_modified(True)
    app._update_project_name_in_title_bar()
    app.custom_title_bar_widget.set_project_name.assert_called_with("test.json*")

def test_show_settings_wip(app, mocker):
    """
    Test that the 'Work in Progress' message is shown for the settings panel.
    """
    mock_qmessagebox = mocker.patch('PySide6.QtWidgets.QMessageBox.information')
    app.show_settings_wip()
    mock_qmessagebox.assert_called_once_with(
        app,
        "Work in Progress",
        "The settings panel is currently under development.",
    )

def test_load_sample_action(app, mocker):
    """
    Test that loading a sample optic works correctly.
    """
    from optiland.samples import CookeTriplet
    mock_load_optic = mocker.patch.object(app.connector, 'load_optic_from_object')
    app._load_sample_action(CookeTriplet)
    mock_load_optic.assert_called_once()
    assert isinstance(mock_load_optic.call_args[0][0], CookeTriplet)

def test_load_sample_action_error(app, mocker):
    """
    Test that an error is handled correctly when loading a sample optic.
    """
    class MockBadOptic:
        def __init__(self):
            raise ValueError("Test error")

    mock_qmessagebox = mocker.patch('PySide6.QtWidgets.QMessageBox.critical')
    app._load_sample_action(MockBadOptic)
    mock_qmessagebox.assert_called_once()
    # Check that the error message contains the sample name and the exception text
    assert "MockBadOptic" in mock_qmessagebox.call_args[0][2]
    assert "Test error" in mock_qmessagebox.call_args[0][2]

def test_animate_dock_toggle(app, mocker):
    """
    Test that the correct animation is triggered when toggling a dock widget.
    """
    mock_animate_show = mocker.patch.object(app, '_animate_dock_show')
    mock_animate_hide = mocker.patch.object(app, '_animate_dock_hide')

    dock = app.panel_manager.lens_editor_dock

    # Test showing the dock
    app.animate_dock_toggle(dock, True)
    mock_animate_show.assert_called_once_with(dock, True, mocker.ANY, 150, mocker.ANY)
    mock_animate_hide.assert_not_called()

    mock_animate_show.reset_mock()

    # Test hiding the dock
    app.animate_dock_toggle(dock, False)
    mock_animate_hide.assert_called_once_with(dock, True, mocker.ANY, 150, mocker.ANY)
    mock_animate_show.assert_not_called()

def test_about_action(app, mocker):
    """
    Test that the about dialog is created and shown correctly.
    """
    mock_dialog_exec = mocker.patch('PySide6.QtWidgets.QDialog.exec')
    mocker.patch('PySide6.QtCore.QPropertyAnimation.start')

    app.about_action()

    assert app.about_dialog is not None
    assert "About Optiland GUI" in app.about_dialog.windowTitle()
    mock_dialog_exec.assert_called_once()
