import pytest
from PySide6.QtWidgets import QWidget, QPushButton
@pytest.fixture
def analysis_panel(app):
    """Returns the AnalysisPanel instance from the main application window."""
    return app.panel_manager.analysis_panel

def test_analysis_panel_creation(analysis_panel):
    """
    Test that the AnalysisPanel widget can be created without errors.
    """
    assert isinstance(analysis_panel, QWidget)
    assert analysis_panel.connector is not None
    assert analysis_panel.analysisTypeCombo is not None
    assert analysis_panel.btnRun is not None
    assert analysis_panel.btnRunAll is not None
    assert analysis_panel.btnStop is not None
    assert analysis_panel.plot_display_frame is not None
    assert analysis_panel.settings_area_widget is not None
    assert analysis_panel.logArea is not None

def test_analysis_type_change(mocker, analysis_panel):
    """
    Test that changing the analysis type updates the settings UI.
    """
    mocker.patch.object(analysis_panel, '_update_settings_ui')

    # Simulate a user changing the analysis type
    analysis_panel.analysisTypeCombo.setCurrentText("Ray Fan")

    analysis_panel._update_settings_ui.assert_called_with("Ray Fan")

def test_run_analysis_button(mocker, analysis_panel):
    """
    Test that clicking the 'Run' button calls the run_analysis_slot.
    """
    mocker.patch.object(analysis_panel, '_execute_analysis', return_value={"name": "Spot Diagram"})

    # Simulate a user clicking the run button
    analysis_panel.btnRun.click()

    analysis_panel._execute_analysis.assert_called_once()
    assert len(analysis_panel.analysis_results_pages) == 1
    assert analysis_panel.current_plot_page_index == 0


def test_settings_widget_creation(analysis_panel):
    """
    Test that the correct settings widgets are created when the analysis type changes.
    """
    # Simulate a user changing the analysis type to "Spot Diagram"
    analysis_panel.analysisTypeCombo.setCurrentText("Spot Diagram")

    # Check that the widgets for the "Spot Diagram" analysis have been created
    expected_widgets = ["num_rings", "distribution", "fields", "wavelengths", "add_airy_disk"]
    for widget_name in expected_widgets:
        assert widget_name in analysis_panel.current_settings_widgets

def test_apply_settings_button(mocker, analysis_panel):
    """
    Test that clicking the 'Apply Settings' button reruns the analysis.
    """
    # Mock the analysis execution and set up an initial analysis page
    mocker.patch.object(analysis_panel, '_execute_analysis', return_value={"name": "Spot Diagram"})
    analysis_panel.analysis_results_pages = [{"name": "Spot Diagram"}]
    analysis_panel.current_plot_page_index = 0

    # Simulate a user clicking the apply settings button
    analysis_panel.btnApplySettings.click()

    analysis_panel._execute_analysis.assert_called_once()

def test_save_settings_button(mocker, analysis_panel):
    """
    Test that clicking the 'Save Settings' button opens a file dialog and saves the settings.
    """
    # Simulate a user changing the analysis type to "Spot Diagram"
    analysis_panel.analysisTypeCombo.setCurrentText("Spot Diagram")

    # Mock the file dialog and the open function
    mocker.patch('PySide6.QtWidgets.QFileDialog.getSaveFileName', return_value=('settings.json', ''))
    mock_open = mocker.patch('builtins.open', mocker.mock_open())

    # Simulate a user clicking the save settings button
    analysis_panel.btnSaveSettings.click()

    # Check that the file dialog was opened and the settings were saved
    mock_open.assert_called_once_with('settings.json', 'w')
    # This is a basic check. A more thorough test would check the contents of the file.

def test_load_settings_button(mocker, analysis_panel):
    """
    Test that clicking the 'Load Settings' button opens a file dialog and loads the settings.
    """
    # Mock the file dialog and the open function
    mocker.patch('PySide6.QtWidgets.QFileDialog.getOpenFileName', return_value=('settings.json', ''))
    settings_json = '{"analysis_name": "Spot Diagram", "constructor_args": {"num_rings": 10}}'
    mock_open = mocker.patch('builtins.open', mocker.mock_open(read_data=settings_json))

    # Simulate a user clicking the load settings button
    analysis_panel.btnLoadSettings.click()

    # Check that the file dialog was opened and the settings were loaded
    mock_open.assert_called_once_with('settings.json')
    assert analysis_panel.analysisTypeCombo.currentText() == "Spot Diagram"
    assert analysis_panel.current_settings_widgets["num_rings"].value() == 10

def test_toggle_settings_panel(analysis_panel, mocker):
    """
    Test that clicking the toggle button calls setVisible on the settings widget.
    """
    # Mock the isVisible method to control the flow of the slot
    mocker.patch.object(analysis_panel.settings_area_widget, 'isVisible', side_effect=[False, True])
    spy = mocker.spy(analysis_panel.settings_area_widget, 'setVisible')

    # Click to show
    analysis_panel.toggleSettingsButton.click()
    spy.assert_called_once_with(True)

    # Click to hide again
    analysis_panel.toggleSettingsButton.click()
    spy.assert_called_with(False)

def test_run_all_analysis_slot(analysis_panel, mocker):
    """
    Test that clicking the 'Run All' button calls the run_all_analysis_slot.
    """
    mocker.patch.object(analysis_panel.logArea, 'append')
    analysis_panel.btnRunAll.click()
    analysis_panel.logArea.append.assert_called_with("Run All: Not yet implemented.")

def test_stop_analysis_slot(analysis_panel, mocker):
    """
    Test that clicking the 'Stop' button calls the stop_analysis_slot.
    """
    mocker.patch.object(analysis_panel.logArea, 'append')
    analysis_panel.btnStop.click()
    analysis_panel.logArea.append.assert_called_with("Stop: Not yet implemented.")

def test_get_value_from_combobox(analysis_panel):
    """
    Test that the _get_value_from_combobox method returns the correct value.
    """
    analysis_panel.analysisTypeCombo.setCurrentText("Ray Fan")
    value = analysis_panel._get_value_from_combobox(
        analysis_panel.analysisTypeCombo, "analysis_type"
    )
    assert value == "Ray Fan"

def test_get_value_from_spinbox(analysis_panel):
    """
    Test that the _get_value_from_spinbox method returns the correct value.
    """
    analysis_panel.analysisTypeCombo.setCurrentText("Spot Diagram")
    analysis_panel.current_settings_widgets["num_rings"].setValue(10)
    value = analysis_panel._get_value_from_spinbox(
        analysis_panel.current_settings_widgets["num_rings"]
    )
    assert value == 10

def test_get_value_from_checkbox(analysis_panel):
    """
    Test that the _get_value_from_checkbox method returns the correct value.
    """
    analysis_panel.analysisTypeCombo.setCurrentText("Spot Diagram")
    analysis_panel.current_settings_widgets["add_airy_disk"].setChecked(True)
    value = analysis_panel._get_value_from_checkbox(
        analysis_panel.current_settings_widgets["add_airy_disk"]
    )
    assert value is True

def test_get_value_from_spinbox_for_rayfan(analysis_panel):
    """
    Test that the _get_value_from_spinbox method returns the correct value for RayFan.
    """
    analysis_panel.analysisTypeCombo.setCurrentText("Ray Fan")
    analysis_panel.current_settings_widgets["num_points"].setValue(10)
    value = analysis_panel._get_value_from_spinbox(
        analysis_panel.current_settings_widgets["num_points"]
    )
    assert value == 10

def test_get_value_from_lineedit_for_fft_mtf(analysis_panel):
    """
    Test that the _get_value_from_lineedit method returns the correct value for FFT MTF.
    """
    analysis_panel.analysisTypeCombo.setCurrentText("FFT MTF")
    # Test integer value
    analysis_panel.current_settings_widgets["max_freq"].setText("100")
    value = analysis_panel._get_value_from_lineedit(
        analysis_panel.current_settings_widgets["max_freq"], "max_freq"
    )
    assert value == 100

    # Test string value
    analysis_panel.current_settings_widgets["max_freq"].setText("cutoff")
    value = analysis_panel._get_value_from_lineedit(
        analysis_panel.current_settings_widgets["max_freq"], "max_freq"
    )
    assert value == "cutoff"

def test_validate_system_for_analysis(analysis_panel, mocker):
    """
    Test that the _validate_system_for_analysis method returns the correct value.
    """
    mocker.patch("PySide6.QtWidgets.QMessageBox.warning")
    # Test with no optic
    assert analysis_panel._validate_system_for_analysis(None) is False
    # Test with no surfaces
    optic = analysis_panel.connector.get_optic()
    optic.surface_group.clear()
    assert analysis_panel._validate_system_for_analysis(optic) is False
    # Test with no wavelengths
    analysis_panel.connector.new_system()
    optic = analysis_panel.connector.get_optic()
    optic.wavelengths.wavelengths.clear()
    assert analysis_panel._validate_system_for_analysis(optic) is False
    # Test with valid optic
    analysis_panel.connector.new_system()
    optic = analysis_panel.connector.get_optic()
    assert analysis_panel._validate_system_for_analysis(optic) is True


def test_real_analysis_execution(analysis_panel):
    """
    Test that a real analysis can be executed and a plot is generated.
    """
    from optiland.samples import CookeTriplet
    # Load a sample optic
    analysis_panel.connector.load_optic_from_object(CookeTriplet())
    # Select an analysis type
    analysis_panel.analysisTypeCombo.setCurrentText("Spot Diagram")
    # Run the analysis
    analysis_panel.btnRun.click()
    # Check that a plot has been generated
    assert len(analysis_panel.analysis_results_pages) == 1
    assert analysis_panel.current_plot_page_index == 0
    # Check for a FigureCanvas widget in the plot container
    canvas_found = False
    for i in range(analysis_panel.plot_container_widget.layout().count()):
        widget = analysis_panel.plot_container_widget.layout().itemAt(i).widget()
        if "FigureCanvas" in widget.__class__.__name__:
            canvas_found = True
            break
    assert canvas_found


def test_plot_navigation(analysis_panel):
    """
    Test that the next and previous plot buttons work correctly.
    """
    from optiland.samples import CookeTriplet
    # Load a sample optic
    analysis_panel.connector.load_optic_from_object(CookeTriplet())
    # Run two analyses
    analysis_panel.analysisTypeCombo.setCurrentText("Spot Diagram")
    analysis_panel.btnRun.click()
    analysis_panel.analysisTypeCombo.setCurrentText("Ray Fan")
    analysis_panel.btnRun.click()
    # Check that there are two plots
    assert len(analysis_panel.analysis_results_pages) == 2
    assert analysis_panel.current_plot_page_index == 1
    # Get the page buttons
    page_buttons = analysis_panel.page_buttons_scroll_area.widget().findChildren(QPushButton)
    assert len(page_buttons) == 2
    # Click the first page button
    page_buttons[0].click()
    assert analysis_panel.current_plot_page_index == 0
    # Click the second page button
    page_buttons[1].click()
    assert analysis_panel.current_plot_page_index == 1


def test_analysis_error_handling(analysis_panel, mocker):
    """
    Test that an error message is displayed if an analysis fails.
    """
    from optiland.samples import CookeTriplet
    # Load a sample optic
    analysis_panel.connector.load_optic_from_object(CookeTriplet())
    # Mock the analysis to raise an exception
    mocker.patch.object(
        analysis_panel, "_execute_analysis", side_effect=Exception("Test Error")
    )
    # Spy on the log area
    spy = mocker.spy(analysis_panel.logArea, "append")
    # Run the analysis
    analysis_panel.btnRun.click()
    # Check that an error message was logged
    assert "Error" in spy.call_args[0][0]
