# optiland_gui/analysis_panel.py
from PySide6.QtWidgets import (QComboBox, QLabel, QPushButton, 
                               QVBoxLayout, QWidget
, QTextEdit, QMessageBox) # Added QTextEdit, QMessageBox
from PySide6.QtCore import Slot
from .optiland_connector import OptilandConnector

# Import necessary Optiland analysis modules
from optiland.analysis import (SpotDiagram, EncircledEnergy, RayFan, YYbar,
                               Distortion, GridDistortion, FieldCurvature,
                               RmsSpotSizeVsField, RmsWavefrontErrorVsField,
                               PupilAberration, IncoherentIrradiance, ThroughFocusSpotDiagram)
# For MTF, we have GeometricMTF and FFTMTF from optiland.mtf
from optiland.mtf import GeometricMTF, FFTMTF



class AnalysisPanel(QWidget):
    ANALYSIS_MAP = {
        "Spot Diagram": SpotDiagram,
        "Encircled Energy": EncircledEnergy,
        "Ray Fan": RayFan,
        "Y-Ybar Diagram": YYbar,
        "Distortion Plot": Distortion,
        "Grid Distortion": GridDistortion,
        "Field Curvature": FieldCurvature,
        "RMS Spot Size vs Field": RmsSpotSizeVsField,
        "RMS Wavefront Error vs Field": RmsWavefrontErrorVsField,
        "Pupil Aberration": PupilAberration,
        # "Incoherent Irradiance": IncoherentIrradiance, # May need more params
        # "Through Focus Spot Diagram": ThroughFocusSpotDiagram, # May need more params
        "Geometric MTF": GeometricMTF,
        "FFT MTF": FFTMTF,
    }

    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Analysis")

        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("Analysis Type:"))
        self.analysisTypeCombo = QComboBox()
        self.analysisTypeCombo.addItems(list(self.ANALYSIS_MAP.keys()))
        self.layout.addWidget(self.analysisTypeCombo)

        self.btnRunAnalysis = QPushButton("Run Analysis")
        self.layout.addWidget(self.btnRunAnalysis)

        self.resultsArea = QTextEdit() # Changed to QTextEdit for more info
        self.resultsArea.setReadOnly(True)
        self.resultsArea.setPlaceholderText("Select an analysis and click 'Run Analysis'. Plots will appear in separate windows.")
        self.layout.addWidget(self.resultsArea, 1)

        self.btnRunAnalysis.clicked.connect(self.run_analysis)

    @Slot()
    def run_analysis(self):
        selected_analysis_name = self.analysisTypeCombo.currentText()
        analysis_class = self.ANALYSIS_MAP.get(selected_analysis_name)
        optic = self.connector.get_optic()

        if not optic or optic.surface_group.num_surfaces <= 2: # Min 3: obj, surf, img
            QMessageBox.warning(self, "Analysis Error", "Cannot run analysis on an empty or minimal system.")
            self.resultsArea.setText("Cannot run analysis: Optic system is not sufficiently defined.")
            return
        if optic.wavelengths.num_wavelengths == 0:
            QMessageBox.warning(self, "Analysis Error", "Optic has no wavelengths defined.")
            self.resultsArea.setText("Cannot run analysis: No wavelengths defined in the optic.")
            return


        if analysis_class:
            self.resultsArea.setText(f"Running {selected_analysis_name}...\nPlots will appear in external windows.")
            try:
                # Most analysis classes take optic, fields, wavelengths.
                # Using "all" for fields and "primary" or "all" for wavelengths as default.
                # Some might need specific parameters (e.g., num_rays, distribution).
                # For simplicity, using common defaults. More UI controls can be added later.
                
                # Default parameters for most analyses
                fields_arg = "all"
                # Use primary wavelength if available, else all, else default from connector
                primary_wl_val = self.connector._get_safe_primary_wavelength_value()
                wavelengths_arg = [primary_wl_val] if primary_wl_val else "all"


                if selected_analysis_name == "Geometric MTF":
                    # GeometricMTF(optic, fields, wavelength, num_rays, distribution, num_points, max_freq, scale)
                    analysis_instance = analysis_class(optic, wavelengths=primary_wl_val) # Uses primary_wavelength string internally
                elif selected_analysis_name == "FFT MTF":
                    # FFTMTF(optic, fields, wavelength, num_rays, grid_size, max_freq)
                    analysis_instance = analysis_class(optic, wavelengths=primary_wl_val)
                elif selected_analysis_name == "Spot Diagram":
                     analysis_instance = analysis_class(optic, fields=fields_arg, wavelengths="all") # SpotDiagram often uses all wavelengths
                else:
                    # Generic instantiation for others
                    analysis_instance = analysis_class(optic, fields=fields_arg, wavelengths=wavelengths_arg)
                
                analysis_instance.view() # This will call plt.show() internally
                self.resultsArea.append(f"{selected_analysis_name} completed.")
                print(f"Analysis Panel: Ran {selected_analysis_name}")

            except Exception as e:
                self.resultsArea.append(f"Error running {selected_analysis_name}: {e}")
                QMessageBox.critical(self, "Analysis Error", f"Error during {selected_analysis_name}:\n{e}")
                print(f"Analysis Panel Error: {e}")
        else:
            self.resultsArea.setText(f"Analysis type '{selected_analysis_name}' not yet implemented.")