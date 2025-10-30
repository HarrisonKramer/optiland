import pytest
from PySide6.QtWidgets import QWidget, QTableWidgetItem
from optiland_gui.lens_editor import LensEditor

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
