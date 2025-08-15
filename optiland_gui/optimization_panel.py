"""Defines the optimization panel for the Optiland GUI.

This module contains the `OptimizationPanel` widget, which provides a user
interface for setting up and running optical system optimizations. This is
currently a placeholder and will be expanded in the future.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from .optiland_connector import OptilandConnector


class OptimizationPanel(QWidget):
    """A widget for configuring and running optical optimizations.

    This panel provides controls for selecting optimization variables, defining
    an objective function, and initiating the optimization process. It is
    connected to the backend via the `OptilandConnector`.

    Note: This is currently a placeholder implementation.

    Attributes:
        connector (OptilandConnector): The connector to the Optiland backend.
    """

    def __init__(self, connector: OptilandConnector, parent=None):
        """Initializes the OptimizationPanel.

        Args:
            connector: The `OptilandConnector` instance for backend communication.
            parent: The parent widget, if any.
        """
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Optimization")

        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("Optimization Variables:"))
        self.variablesList = QListWidget()
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
        """Starts the optimization process.

        This is a placeholder method that will eventually trigger the optimization
        engine in the Optiland backend.
        """
        self.resultsArea.setText(
            "Starting optimization...\n(Placeholder - connect to Optiland backend)"
        )
        print("Optimization Panel: Starting optimization")
