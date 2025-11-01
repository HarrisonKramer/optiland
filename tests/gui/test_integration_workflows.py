"""
High-level integration and workflow tests for the Optiland GUI.

This module contains end-to-end tests that simulate real user workflows
and verify the contracts between the GUI and the `optiland` core library.
These tests are designed to catch regressions that might be missed by
atomic component tests.
"""

import pytest
from optiland.samples import CookeTriplet

def test_full_user_workflow(app):
    """
    Test a full end-to-end user workflow:
    1. Load a sample system.
    2. Verify the UI is populated.
    3. Change a system property.
    4. Run an analysis.
    """
    # 1. Load a sample system
    optic = CookeTriplet()
    app.connector.load_optic_from_object(optic)

    # 2. Verify the UI is populated
    lens_editor = app.panel_manager.lens_editor
    assert lens_editor.tableWidget.rowCount() == optic.surface_group.num_surfaces

    # 3. Change a system property
    properties_panel = app.panel_manager.system_properties
    aperture_editor = properties_panel.apertureEditor
    aperture_editor.spnApertureValue.setValue(12.0)
    aperture_editor.btnApplyAperture.click()

    # 4. Run an analysis
    analysis_panel = app.panel_manager.analysis_panel
    analysis_panel.analysisTypeCombo.setCurrentText("Spot Diagram")
    analysis_panel.btnRun.click()

    # Assert that a plot was generated
    assert len(analysis_panel.analysis_results_pages) > 0
    assert analysis_panel.active_mpl_canvas_widget is not None

from optiland_gui.analysis_panel import AnalysisPanel

@pytest.mark.parametrize("analysis_name", AnalysisPanel.ANALYSIS_MAP.keys())
def test_analysis_contract(app, analysis_name):
    """
    Test that all registered analyses can be instantiated and run without errors.
    This acts as a contract test to ensure GUI compatibility with the analysis backend.
    """
    # Load a standard optical system
    optic = CookeTriplet()
    app.connector.load_optic_from_object(optic)

    # Get the analysis panel
    analysis_panel = app.panel_manager.analysis_panel

    # Select and run the analysis
    analysis_panel.analysisTypeCombo.setCurrentText(analysis_name)
    analysis_panel.btnRun.click()

    # The test passes if no exceptions were raised during the run
    assert len(analysis_panel.analysis_results_pages) > 0
