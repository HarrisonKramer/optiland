# optiland_gui/optimization_panel.py
from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .optiland_connector import OptilandConnector


class OptimizationPanel(QWidget):
    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Optimization")

        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("Optimization Variables:"))
        self.variablesList = QListWidget()  # Placeholder for variable selection
        self.variablesList.addItems(["Radius S2", "Thickness S1", "Conic S3"])
        self.layout.addWidget(self.variablesList)

        self.layout.addWidget(QLabel("Objective (e.g., Minimize RMS Spot Size):"))
        self.objectiveEdit = QLineEdit("RMS Spot Size")
        self.layout.addWidget(self.objectiveEdit)

        self.btnStartOptimization = QPushButton("Start Optimization")
        self.layout.addWidget(self.btnStartOptimization)

        self.resultsArea = QLabel("Optimization progress/results will appear here.")
        self.resultsArea.setWordWrap(True)
        self.layout.addWidget(self.resultsArea, 1)

        self.btnStartOptimization.clicked.connect(self.start_optimization)

    def start_optimization(self):
        # Placeholder: Trigger Optiland optimization routines
        self.resultsArea.setText(
            "Starting optimization...\n(Placeholder - connect to Optiland backend)"
        )
        print("Optimization Panel: Starting optimization")
