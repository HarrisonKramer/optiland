import pytest
from PySide6.QtWidgets import QWidget
@pytest.fixture
def analysis_panel(app):
    """Returns the AnalysisPanel instance from the main application window."""
    return app.panel_manager.analysis_panel

def test_analysis_panel_creation(analysis_panel):
    """
    Test that the AnalysisPanel widget can be created without errors.
    """
    assert isinstance(analysis_panel, QWidget)

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
