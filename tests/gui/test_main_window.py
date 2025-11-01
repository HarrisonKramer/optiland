from optiland_gui.main_window import MainWindow

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
