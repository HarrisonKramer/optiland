import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from optiland_gui.system_properties_panel import (
    ApertureEditor,
    FieldsEditor,
    SystemPropertiesPanel,
    WavelengthsEditor,
)

@pytest.fixture
def properties_panel(app):
    """Returns the SystemPropertiesPanel instance from the main application window."""
    return app.panel_manager.system_properties

def test_system_properties_panel_creation(properties_panel):
    """
    Test that the SystemPropertiesPanel and its child editors are created.
    """
    assert isinstance(properties_panel, SystemPropertiesPanel)
    assert isinstance(properties_panel.apertureEditor, ApertureEditor)
    assert isinstance(properties_panel.fieldsEditor, FieldsEditor)
    assert isinstance(properties_panel.wavelengthsEditor, WavelengthsEditor)

def test_navigation_tree_switching(properties_panel):
    """
    Test that clicking an item in the navigation tree switches the stacked widget.
    """
    # Find the "Fields" item in the navigation tree
    fields_item = properties_panel.navTree.findItems("Fields", Qt.MatchExactly)[0]

    # Simulate a click on the "Fields" item
    properties_panel.navTree.itemClicked.emit(fields_item, 0)

    # Check that the stacked widget has switched to the FieldsEditor
    assert properties_panel.stackedWidget.currentWidget() == properties_panel.fieldsEditor

def test_aperture_editor_changes(mocker, properties_panel):
    """
    Test that clicking the 'Apply' button calls the connector with the correct values.
    """
    # Get the aperture editor
    aperture_editor = properties_panel.apertureEditor

    # Mock the set_aperture method on the Optic object
    mock_set_aperture = mocker.patch.object(properties_panel.connector.get_optic(), 'set_aperture')

    # Disconnect the reactive signals to isolate the button's functionality
    aperture_editor.cmbApertureType.currentTextChanged.disconnect(aperture_editor.apply_aperture_changes)
    aperture_editor.spnApertureValue.valueChanged.disconnect(aperture_editor.apply_aperture_changes)

    # Simulate a user changing the aperture type and value
    aperture_editor.cmbApertureType.setCurrentText("imageFNO")
    aperture_editor.spnApertureValue.setValue(5.6)

    # Ensure no calls happened yet
    mock_set_aperture.assert_not_called()

    # Simulate a click on the apply button
    aperture_editor.btnApplyAperture.click()

    # Check that the set_aperture method was called once with the correct arguments
    mock_set_aperture.assert_called_once_with("imageFNO", 5.6)

def test_fields_editor_add_field(mocker, properties_panel):
    """
    Test that clicking the 'Add Field' button calls the connector.
    """
    # Get the fields editor
    fields_editor = properties_panel.fieldsEditor

    # Mock the add_field method on the Optic object
    mock_add_field = mocker.patch.object(properties_panel.connector.get_optic(), 'add_field')

    # Simulate a click on the add field button
    fields_editor.btnAddField.click()

    # Check that the add_field method was called
    mock_add_field.assert_called_once()

def test_fields_editor_remove_field(mocker, properties_panel):
    """
    Test that clicking the 'Remove Field' button removes the selected field.
    """
    # Get the fields editor
    fields_editor = properties_panel.fieldsEditor

    # Prepare a mock field and a real list to hold it
    optic = properties_panel.connector.get_optic()
    mock_field = mocker.MagicMock()
    mock_field.x, mock_field.y, mock_field.vx, mock_field.vy = 0, 1, 0, 0
    field_list = [mock_field]

    # Mock the fields attribute of the optic object
    mocker.patch.object(optic, 'fields')
    optic.fields.num_fields = 1
    optic.fields.fields = field_list

    # Load the data into the table
    fields_editor.load_data()

    # Select the first row in the UI
    fields_editor.tableFields.selectRow(0)

    # Simulate a click on the remove field button
    fields_editor.btnRemoveField.click()

    # Assert that the item was removed from the original list
    assert len(field_list) == 0

def test_fields_editor_apply_changes(mocker, properties_panel):
    """
    Test that clicking the 'Apply Field Changes' button updates the field data.
    """
    # Get the fields editor
    fields_editor = properties_panel.fieldsEditor

    # Prepare a mock field object
    optic = properties_panel.connector.get_optic()
    mock_field = mocker.MagicMock()
    mock_field.x, mock_field.y, mock_field.vx, mock_field.vy = 0, 1, 0, 0

    # Mock the fields attribute of the optic object
    mocker.patch.object(optic, 'fields')
    optic.fields.num_fields = 1
    optic.fields.fields = [mock_field]

    # Load the data into the table
    fields_editor.load_data()

    # Simulate a user editing the table
    fields_editor.tableFields.item(0, 1).setText("0.8")

    # Simulate a click on the apply button
    fields_editor.btnApplyFields.click()

    # Assert that the field object was updated
    assert mock_field.y == 0.8

def test_wavelengths_editor_add_wavelength(mocker, properties_panel):
    """
    Test that clicking the 'Add Wavelength' button calls the connector.
    """
    # Get the wavelengths editor
    wavelengths_editor = properties_panel.wavelengthsEditor

    # Mock the add_wavelength method on the Optic object
    mock_add_wavelength = mocker.patch.object(properties_panel.connector.get_optic(), 'add_wavelength')

    # Simulate a click on the add wavelength button
    wavelengths_editor.btnAddWavelength.click()

    # Check that the add_wavelength method was called
    mock_add_wavelength.assert_called_once()

def test_wavelengths_editor_remove_wavelength(mocker, properties_panel):
    """
    Test that clicking the 'Remove Wavelength' button removes the selected item.
    """
    wavelengths_editor = properties_panel.wavelengthsEditor
    optic = properties_panel.connector.get_optic()

    wl1 = mocker.MagicMock(value=0.587, is_primary=True)
    wl2 = mocker.MagicMock(value=0.656, is_primary=False)
    wavelength_list = [wl1, wl2]

    mocker.patch.object(optic, 'wavelengths')
    optic.wavelengths.num_wavelengths = 2
    optic.wavelengths.wavelengths = wavelength_list

    wavelengths_editor.load_data()
    wavelengths_editor.tableWavelengths.selectRow(1)
    wavelengths_editor.btnRemoveWavelength.click()

    assert len(wavelength_list) == 1
    assert wavelength_list[0].value == 0.587

def test_wavelengths_editor_set_primary(mocker, properties_panel):
    """
    Test that clicking the 'Set Selected as Primary' button updates the primary status.
    """
    wavelengths_editor = properties_panel.wavelengthsEditor
    optic = properties_panel.connector.get_optic()

    wl1 = mocker.MagicMock(value=0.587, is_primary=True, _unit="um")
    wl2 = mocker.MagicMock(value=0.656, is_primary=False, _unit="um")

    mocker.patch.object(optic, 'wavelengths')
    optic.wavelengths.num_wavelengths = 2
    optic.wavelengths.wavelengths = [wl1, wl2]

    wavelengths_editor.load_data()
    wavelengths_editor.tableWavelengths.selectRow(1)
    wavelengths_editor.btnSetPrimary.click()

    assert not wl1.is_primary
    assert wl2.is_primary

def test_wavelengths_editor_apply_changes(mocker, properties_panel):
    """
    Test that clicking 'Apply Wavelength Changes' updates the wavelength value.
    """
    wavelengths_editor = properties_panel.wavelengthsEditor
    optic = properties_panel.connector.get_optic()

    wl1 = mocker.MagicMock(value=0.587, is_primary=True, _unit="um")

    mocker.patch.object(optic, 'wavelengths')
    optic.wavelengths.num_wavelengths = 1
    optic.wavelengths.wavelengths = [wl1]

    wavelengths_editor.load_data()
    wavelengths_editor.tableWavelengths.item(0, 0).setText("0.6328")
    wavelengths_editor.btnApplyWavelengths.click()

    assert wl1._value == 0.6328
