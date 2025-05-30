# optiland_gui/main_window.py
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
)

from .analysis_panel import AnalysisPanel
from .lens_editor import LensEditor
from .optiland_connector import OptilandConnector
from .optimization_panel import OptimizationPanel
from .viewer_panel import ViewerPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optiland GUI")
        self.setGeometry(100, 100, 1600, 900)

        self.connector = OptilandConnector()

        # --- Create Panels ---
        self.lensEditor = LensEditor(self.connector)
        self.viewerPanel = ViewerPanel(
            self.connector
        )  # This will be the central widget
        self.analysisPanel = AnalysisPanel(self.connector)
        self.optimizationPanel = OptimizationPanel(self.connector)

        # Set ViewerPanel as the central widget
        self.setCentralWidget(self.viewerPanel)

        # --- Dock Widgets ---
        # Lens Editor Dock
        self.lensEditorDock = QDockWidget("Lens Editor", self)
        self.lensEditorDock.setWidget(self.lensEditor)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.lensEditorDock)

        # Analysis Panel Dock
        self.analysisPanelDock = QDockWidget("Analysis", self)
        self.analysisPanelDock.setWidget(self.analysisPanel)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.analysisPanelDock
        )

        # Optimization Panel Dock
        self.optimizationPanelDock = QDockWidget("Optimization", self)
        self.optimizationPanelDock.setWidget(self.optimizationPanel)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.optimizationPanelDock
        )

        # Tabify right docks if desired
        self.tabifyDockWidget(self.analysisPanelDock, self.optimizationPanelDock)
        self.analysisPanelDock.raise_()  # Bring analysis to front initially

        self._create_actions()
        self._create_menu_bar()
        self._create_tool_bars()  # Optional

        # Load initial style (optional)
        # self.load_stylesheet("resources/styles.qss")

    def _create_actions(self):
        # File actions
        self.newAction = QAction(
            "&New System", self, shortcut=QKeySequence.New, triggered=self.new_system
        )
        self.openAction = QAction(
            "&Open System...",
            self,
            shortcut=QKeySequence.Open,
            triggered=self.open_system,
        )
        self.saveAction = QAction(
            "&Save System", self, shortcut=QKeySequence.Save, triggered=self.save_system
        )
        self.saveAsAction = QAction(
            "Save System &As...",
            self,
            shortcut=QKeySequence.SaveAs,
            triggered=self.save_system_as,
        )
        self.exitAction = QAction(
            "E&xit", self, shortcut=QKeySequence.Quit, triggered=self.close
        )

        # View actions for toggling docks
        self.toggleLensEditorAction = self.lensEditorDock.toggleViewAction()
        self.toggleLensEditorAction.setText("Toggle Lens &Editor")
        self.toggleAnalysisPanelAction = self.analysisPanelDock.toggleViewAction()
        self.toggleAnalysisPanelAction.setText("Toggle &Analysis Panel")
        self.toggleOptimizationPanelAction = (
            self.optimizationPanelDock.toggleViewAction()
        )
        self.toggleOptimizationPanelAction.setText("Toggle &Optimization Panel")

        # Help actions
        self.aboutAction = QAction("&About Optiland GUI", self, triggered=self.about)

    def _create_menu_bar(self):
        menuBar = self.menuBar()

        # File Menu
        fileMenu = menuBar.addMenu("&File")
        fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.saveAction)
        fileMenu.addAction(self.saveAsAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)

        # View Menu
        viewMenu = menuBar.addMenu("&View")
        viewMenu.addAction(self.toggleLensEditorAction)
        viewMenu.addAction(self.toggleAnalysisPanelAction)
        viewMenu.addAction(self.toggleOptimizationPanelAction)

        # Run Menu (Placeholder)
        # runMenu = menuBar.addMenu("&Run")
        # Add actions for Run Raytrace, Run Optimization etc.

        # Help Menu
        helpMenu = menuBar.addMenu("&Help")
        helpMenu.addAction(self.aboutAction)

    def _create_tool_bars(self):
        # File Toolbar (Optional)
        fileToolBar = self.addToolBar("File")
        fileToolBar.addAction(self.newAction)
        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.saveAction)
        # Add more toolbar actions as needed

    def load_stylesheet(self, filepath):
        try:
            with open(filepath) as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print(f"Stylesheet not found: {filepath}")
        except Exception as e:
            print(f"Error loading stylesheet: {e}")

    @Slot()
    def new_system(self):
        # Placeholder: Reinitialize or clear the current Optic object
        self.connector._optic = self.connector.DummyOptic(
            "New Untitled System"
        )  # Re-init with dummy
        self.connector.opticChanged.emit()
        print("Main Window: New System triggered")

    @Slot()
    def open_system(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Optiland System",
            "",
            "Optiland Files (*.json *.yaml);;All Files (*)",
        )
        if filepath:
            self.connector.load_optic_from_file(filepath)
            print(f"Main Window: Open System triggered - {filepath}")

    @Slot()
    def save_system(self):
        # Placeholder: if current filepath known, save, else save_as
        print("Main Window: Save System triggered (placeholder)")
        # self.connector.save_optic_to_file("current_file.json") # Example

    @Slot()
    def save_system_as(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Optiland System As...",
            "",
            "Optiland Files (*.json *.yaml);;All Files (*)",
        )
        if filepath:
            self.connector.save_optic_to_file(filepath)
            print(f"Main Window: Save System As triggered - {filepath}")

    @Slot()
    def about(self):
        QMessageBox.about(
            self,
            "About Optiland GUI",
            "<p><b>Optiland GUI</b></p>"
            "<p>A modern interface for the Optiland optical simulation package.</p>"
            "<p>Version: 0.1 (Alpha)</p>"
            "<p>Built with PySide6.</p>",
        )

    def closeEvent(self, event):
        # Add any cleanup or "are you sure?" dialogs here
        print("Main Window: Closing application.")
        event.accept()
