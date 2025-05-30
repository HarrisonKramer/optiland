# optiland_gui/analysis_panel.py
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget

from .optiland_connector import OptilandConnector


class AnalysisPanel(QWidget):
    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Analysis")

        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("Analysis Type:"))
        self.analysisTypeCombo = QComboBox()
        self.analysisTypeCombo.addItems(
            ["Spot Diagram", "MTF", "Field Curvature", "Distortion"]
        )
        self.layout.addWidget(self.analysisTypeCombo)

        self.btnRunAnalysis = QPushButton("Run Analysis")
        self.layout.addWidget(self.btnRunAnalysis)

        self.resultsArea = QLabel("Analysis results will appear here.")
        self.resultsArea.setWordWrap(True)
        self.layout.addWidget(self.resultsArea, 1)  # Add stretch factor

        self.btnRunAnalysis.clicked.connect(self.run_analysis)

    def run_analysis(self):
        analysis_type = self.analysisTypeCombo.currentText()
        # Placeholder: In a real app, this would trigger Optiland analysis
        # and display results (e.g., a Matplotlib plot or text)
        self.resultsArea.setText(
            f"Running {analysis_type}...\n(Placeholder - connect to Optiland backend)"
        )
        print(f"Analysis Panel: Running {analysis_type}")
