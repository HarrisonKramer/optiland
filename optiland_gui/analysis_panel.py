# optiland_gui/analysis_panel.py
import inspect  # Added inspect module

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,  # Added QTextEdit, QMessageBox
    QVBoxLayout,
    QWidget,
)

# Import necessary Optiland analysis modules
from optiland.analysis import (
    Distortion,
    EncircledEnergy,
    FieldCurvature,
    GridDistortion,
    # IncoherentIrradiance,
    PupilAberration,
    RayFan,
    RmsSpotSizeVsField,
    RmsWavefrontErrorVsField,
    SpotDiagram,
    # ThroughFocusSpotDiagram,
    YYbar,
)

# For MTF, we have GeometricMTF and FFTMTF from optiland.mtf
from optiland.mtf import FFTMTF, GeometricMTF

from .optiland_connector import OptilandConnector


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

        self.resultsArea = QTextEdit()  # Changed to QTextEdit for more info
        self.resultsArea.setReadOnly(True)
        self.resultsArea.setPlaceholderText(
            "Select an analysis and click 'Run Analysis'. "
            "Plots will appear in separate windows."
        )
        self.layout.addWidget(self.resultsArea, 1)

        self.btnRunAnalysis.clicked.connect(self.run_analysis)

    @Slot()
    def run_analysis(self):
        selected_analysis_name = self.analysisTypeCombo.currentText()
        analysis_class = self.ANALYSIS_MAP.get(selected_analysis_name)
        optic = self.connector.get_optic()

        if not optic or optic.surface_group.num_surfaces <= 2:  # Min 3: obj, surf, img
            QMessageBox.warning(
                self,
                "Analysis Error",
                "Cannot run analysis on an empty or minimal system.",
            )
            self.resultsArea.setText(
                "Cannot run analysis: Optic system is not sufficiently defined."
            )
            return
        if optic.wavelengths.num_wavelengths == 0:
            QMessageBox.warning(
                self, "Analysis Error", "Optic has no wavelengths defined."
            )
            self.resultsArea.setText(
                "Cannot run analysis: No wavelengths defined in the optic."
            )
            return

        if analysis_class:
            self.resultsArea.setText(
                f"Running {selected_analysis_name}...\nPlots will appear "
                f"in external windows."
            )
            try:
                primary_wl_val = self.connector._get_safe_primary_wavelength_value()

                sig = inspect.signature(analysis_class.__init__)
                constructor_args = {}

                for param in sig.parameters.values():
                    if param.name == "self":
                        continue
                    if param.name in ["optic", "optical_system"]:
                        constructor_args[param.name] = optic
                    elif param.name == "fields":
                        constructor_args[param.name] = "all"
                    elif (
                        param.name == "wavelengths"
                    ):  # Expects a list or "all" or "primary"
                        if primary_wl_val is not None:
                            constructor_args[param.name] = [primary_wl_val]
                        elif param.default != inspect.Parameter.empty:
                            constructor_args[param.name] = (
                                param.default
                            )  # Use class default if any
                        else:
                            constructor_args[param.name] = "all"  # Fallback
                    elif (
                        param.name == "wavelength"
                    ):  # Expects a single value or "primary"
                        if primary_wl_val is not None:
                            constructor_args[param.name] = primary_wl_val
                        elif param.default != inspect.Parameter.empty:
                            constructor_args[param.name] = param.default
                        else:
                            constructor_args[param.name] = (
                                "primary"  # Fallback for single wavelength
                            )
                    elif param.name == "num_rays":
                        # SpotDiagram, GeometricMTF, FFTMTF might use this
                        if selected_analysis_name == "Ray Fan":
                            constructor_args[param.name] = 7
                        else:  # SpotDiagram, MTF
                            constructor_args[param.name] = 24
                    elif param.name == "distribution":
                        if selected_analysis_name == "Ray Fan":
                            constructor_args[param.name] = "line_y"
                        else:  # SpotDiagram, MTF
                            constructor_args[param.name] = "grid"
                    elif param.name == "num_points":  # For MTF
                        constructor_args[param.name] = 50
                    elif param.name == "max_freq":  # For MTF
                        constructor_args[param.name] = 100
                    elif param.name == "grid_size":  # For FFT MTF
                        constructor_args[param.name] = 10
                    elif param.name == "scale":  # For some plots like MTF
                        constructor_args[param.name] = "linear"
                    elif param.name == "pupil_points":  # For PupilAberration
                        constructor_args[param.name] = 32
                    # Add other common parameters if needed, or rely on their defaults
                    # in the class. If a parameter is mandatory and not covered, an
                    # error will still occur at instantiation

                self.resultsArea.append(f"Using arguments: {constructor_args}")
                analysis_instance = analysis_class(**constructor_args)
                analysis_instance.view()  # This will call plt.show() internally
                self.resultsArea.append(f"{selected_analysis_name} completed.")
                print(
                    f"Analysis Panel: Ran {selected_analysis_name} with "
                    f"args: {constructor_args}"
                )

            except Exception as e:
                self.resultsArea.append(f"Error running {selected_analysis_name}: {e}")
                QMessageBox.critical(
                    self,
                    "Analysis Error",
                    f"Error during {selected_analysis_name}:\n{e}",
                )
                print(f"Analysis Panel Error: {e}")
        else:
            self.resultsArea.setText(
                f"Analysis type '{selected_analysis_name}' not yet implemented."
            )
