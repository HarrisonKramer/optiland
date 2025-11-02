import pytest
from PySide6.QtWidgets import QWidget, QTableWidgetItem

@pytest.fixture
def lens_editor(app):
    """Returns the LensEditor instance from the main application window."""
    return app.panel_manager.lens_editor

def test_lens_editor_creation(lens_editor):
    """
    Test that the LensEditor widget can be created without errors.
    """
    assert isinstance(lens_editor, QWidget)

def test_add_surface_button(mocker, lens_editor):
    """
    Test that clicking the 'Add Surface' button calls the connector's
    add_surface method.
    """
    mocker.patch.object(lens_editor.connector, 'add_surface')
    lens_editor.btnAddSurface.click()
    lens_editor.connector.add_surface.assert_called_once()

def test_remove_surface_button(mocker, lens_editor):
    """
    Test that clicking the 'Remove Surface' button calls the connector's
    remove_surface method.
    """
    # Select a row to be removed
    lens_editor.tableWidget.selectRow(1)
    mocker.patch.object(lens_editor.connector, 'remove_surface')
    lens_editor.btnRemoveSurface.click()
    lens_editor.connector.remove_surface.assert_called_once()

def test_cell_data_change(mocker, lens_editor):
    """
    Test that editing a cell in the table triggers the connector's
    set_surface_data method.
    """
    mocker.patch.object(lens_editor.connector, 'set_surface_data')

    # Simulate a user editing a cell
    item = QTableWidgetItem("100.0")
    lens_editor.tableWidget.setItem(1, 2, item) # Row 1, Column 2 (Radius)

    # The itemChanged signal is emitted, which should call the handler
    lens_editor.on_item_changed_handler(item)

    lens_editor.connector.set_surface_data.assert_called_with(1, 2, "100.0")

def test_toggle_properties_widget(mocker, lens_editor):
    """
    Test that the properties widget is correctly shown and hidden.
    """
    mocker.patch.object(lens_editor, 'load_data')

    # Initially, no properties widget is open
    assert lens_editor.open_prop_source_row == -1

    # Toggle to show the properties for row 1
    lens_editor.toggle_properties_widget(1)
    assert lens_editor.open_prop_source_row == 1
    lens_editor.load_data.assert_called_once()

    # Toggle again to hide the properties
    lens_editor.toggle_properties_widget(1)
    assert lens_editor.open_prop_source_row == -1
    assert lens_editor.load_data.call_count == 2


def test_surface_properties_widget_apply_changes(mocker, app):
    """
    Test that the apply_changes method correctly gathers and sends data.
    """
    from optiland_gui.lens_editor import SurfacePropertiesWidget
    connector = app.panel_manager.lens_editor.connector
    row = 1

    mocker.patch.object(connector, 'get_surface_geometry_params', return_value={'conic': 0.0, 'radius': 100.0})
    mock_set_params = mocker.patch.object(connector, 'set_surface_geometry_params')

    widget = SurfacePropertiesWidget(row, connector)

    widget.input_widgets['conic'].setText("-0.5")
    widget.input_widgets['radius'].setText("95.0")

    widget.apply_changes()

    expected_params = {'conic': '-0.5', 'radius': '95.0'}
    mock_set_params.assert_called_once_with(row, expected_params)


def test_surface_properties_widget_no_params(mocker, app):
    """
    Test that the properties widget is created correctly when there are no
    extra parameters for the surface.
    """
    from optiland_gui.lens_editor import SurfacePropertiesWidget
    from PySide6.QtWidgets import QLabel

    connector = app.panel_manager.lens_editor.connector
    row = 1

    # Mock the connector to return no parameters
    mocker.patch.object(connector, 'get_surface_geometry_params', return_value={})

    widget = SurfacePropertiesWidget(row, connector)

    # Verify that a label indicating no properties is shown
    labels = widget.findChildren(QLabel)
    assert len(labels) == 1
    assert "No additional properties" in labels[0].text()


def test_surface_type_widget_type_selected(mocker, qtbot, app):
    """
    Test that selecting a type from the dropdown emits the correct signal.
    """
    from optiland_gui.lens_editor import SurfaceTypeWidget

    mock_connector = mocker.MagicMock()
    mock_connector.get_available_surface_types.return_value = ['standard', 'evenasphere']

    current_type_info = {'display_text': 'Standard', 'is_changeable': True, 'has_extra_params': False}

    widget = SurfaceTypeWidget(1, current_type_info, mock_connector, parent=app)

    with qtbot.waitSignal(widget.surfaceTypeChanged) as blocker:
        action = next(a for a in widget.surface_menu.actions() if a.text().lower() == 'evenasphere')
        action.trigger()

    assert blocker.args == ['evenasphere']


def test_surface_type_widget_text_changed_valid(mocker, qtbot, app):
    """
    Test that a valid text change in the line edit emits the correct signal.
    """
    from optiland_gui.lens_editor import SurfaceTypeWidget

    mock_connector = mocker.MagicMock()
    mock_connector.get_available_surface_types.return_value = ['standard', 'evenasphere']

    current_type_info = {'display_text': 'Standard', 'is_changeable': True, 'has_extra_params': False}

    widget = SurfaceTypeWidget(1, current_type_info, mock_connector, parent=app)

    with qtbot.waitSignal(widget.surfaceTypeChanged) as blocker:
        widget.type_edit.setText('evenasphere')
        widget.type_edit.editingFinished.emit()

    assert blocker.args == ['evenasphere']


def test_surface_type_widget_text_changed_invalid(mocker, app):
    """
    Test that an invalid text change reverts the text and emits no signal.
    """
    from optiland_gui.lens_editor import SurfaceTypeWidget

    mock_connector = mocker.MagicMock()
    mock_connector.get_available_surface_types.return_value = ['standard', 'evenasphere']
    mock_connector.get_surface_type_info.return_value = {'display_text': 'Standard'}

    current_type_info = {'display_text': 'Standard', 'is_changeable': True, 'has_extra_params': False}

    widget = SurfaceTypeWidget(1, current_type_info, mock_connector, parent=app)

    mock_slot = mocker.Mock()
    widget.surfaceTypeChanged.connect(mock_slot)

    widget.type_edit.setText('invalid_type')
    widget.type_edit.editingFinished.emit()

    mock_slot.assert_not_called()
    assert widget.type_edit.text() == 'Standard'
