# optiland_gui/main_window.py
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
)

from .lens_editor import LensEditor
from .viewer_panel import ViewerPanel
from .analysis_panel import AnalysisPanel
from .optimization_panel import OptimizationPanel
from .optiland_connector import OptilandConnector
from .system_properties_panel import SystemPropertiesPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optiland GUI")
        self.setGeometry(100, 100, 1600, 900)

        self.connector = OptilandConnector()

        # --- Create Panels ---
        self.lensEditor = LensEditor(self.connector)
        self.viewerPanel = ViewerPanel(self.connector) 
        self.analysisPanel = AnalysisPanel(self.connector) 
        self.optimizationPanel = OptimizationPanel(self.connector)
        self.systemPropertiesPanel = SystemPropertiesPanel(self.connector)

        self.setCentralWidget(self.viewerPanel)

        # --- Dock Widgets ---
        self.lensEditorDock = QDockWidget("Lens Data Editor", self)
        self.lensEditorDock.setWidget(self.lensEditor)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.lensEditorDock)

        self.systemPropertiesDock = QDockWidget("System Properties", self) 
        self.systemPropertiesDock.setWidget(self.systemPropertiesPanel)
        self.tabifyDockWidget(self.lensEditorDock, self.systemPropertiesDock)
        self.lensEditorDock.raise_() 

        self.analysisPanelDock = QDockWidget("Analysis", self)
        self.analysisPanelDock.setWidget(self.analysisPanel)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.analysisPanelDock
        )

        self.optimizationPanelDock = QDockWidget("Optimization", self)
        self.optimizationPanelDock.setWidget(self.optimizationPanel)
        self.tabifyDockWidget(self.analysisPanelDock, self.optimizationPanelDock)
        self.analysisPanelDock.raise_()

        self._create_actions()
        self._create_menu_bar()
        self._create_tool_bars() 

        # self.load_stylesheet("resources/styles.qss")

    def _create_actions(self):
        # File actions
        self.newAction = QAction("&New System", self, shortcut=QKeySequence.New, triggered=self.new_system_action) # Renamed slot
        self.openAction = QAction("&Open System...", self, shortcut=QKeySequence.Open, triggered=self.open_system_action) # Renamed slot
        self.saveAction = QAction("&Save System", self, shortcut=QKeySequence.Save, triggered=self.save_system_action) # Renamed slot
        self.saveAsAction = QAction("Save System &As...", self, shortcut=QKeySequence.SaveAs, triggered=self.save_system_as_action) # Renamed slot
        self.exitAction = QAction("E&xit", self, shortcut=QKeySequence.Quit, triggered=self.close)

        # View actions for toggling docks
        self.toggleLensEditorAction = self.lensEditorDock.toggleViewAction()
        self.toggleLensEditorAction.setText("Toggle Lens &Editor")
        self.toggleSystemPropertiesAction = self.systemPropertiesDock.toggleViewAction() 
        self.toggleSystemPropertiesAction.setText("Toggle System &Properties")
        self.toggleAnalysisPanelAction = self.analysisPanelDock.toggleViewAction()
        self.toggleAnalysisPanelAction.setText("Toggle &Analysis Panel")
        self.toggleOptimizationPanelAction = (
            self.optimizationPanelDock.toggleViewAction()
        )
        self.toggleOptimizationPanelAction.setText("Toggle &Optimization Panel")

        # Help actions
        self.aboutAction = QAction("&About Optiland GUI", self, triggered=self.about_action) # Renamed slot

    def _create_menu_bar(self):
        menuBar = self.menuBar()

        fileMenu = menuBar.addMenu("&File")
        fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.saveAction)
        fileMenu.addAction(self.saveAsAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)

        viewMenu = menuBar.addMenu("&View")
        viewMenu.addAction(self.toggleLensEditorAction)
        viewMenu.addAction(self.toggleSystemPropertiesAction) 
        viewMenu.addAction(self.toggleAnalysisPanelAction)
        viewMenu.addAction(self.toggleOptimizationPanelAction)
        
        runMenu = menuBar.addMenu("&Run")

        helpMenu = menuBar.addMenu("&Help")
        helpMenu.addAction(self.aboutAction)

    def _create_tool_bars(self):
        fileToolBar = self.addToolBar("File")
        fileToolBar.addAction(self.newAction)
        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.saveAction)

    def load_stylesheet(self, filepath):
        try:
            with open(filepath) as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print(f"Stylesheet not found: {filepath}")
        except Exception as e:
            print(f"Error loading stylesheet: {e}")

    @Slot()
    def new_system_action(self): # Renamed to avoid conflict with any potential internal 'new_system'
        self.connector.new_system() # Calls the connector's method
        print("Main Window: New System action triggered")

    @Slot()
    def open_system_action(self): # Renamed
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Optiland System", "", "Optiland JSON Files (*.json);;All Files (*)")
        if filepath:
            self.connector.load_optic_from_file(filepath)
            print(f"Main Window: Open System action triggered - {filepath}")

    @Slot()
    def save_system_action(self): # Renamed
        current_path = self.connector.get_current_filepath()
        if current_path:
            self.connector.save_optic_to_file(current_path)
            print(f"Main Window: Save System action triggered - {current_path}")
        else:
            self.save_system_as_action() # If no current path, behave like Save As

    @Slot()
    def save_system_as_action(self): # Renamed
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Optiland System As...", "", "Optiland JSON Files (*.json);;All Files (*)")
        if filepath:
            # Ensure the filepath ends with .json if the filter selected it
            if not filepath.lower().endswith(".json") and "(*.json)" in _.split(";;")[0] :
                 filepath += ".json"
            self.connector.save_optic_to_file(filepath)
            print(f"Main Window: Save System As action triggered - {filepath}")

    @Slot()
    def about_action(self): # Renamed
        QMessageBox.about(
            self,
            "About Optiland GUI",
            "<p><b>Optiland GUI</b></p>"
            "<p>A modern interface for the Optiland optical simulation package.</p>"
            "<p>Version: 0.1.1 (Alpha)</p>" # Slightly incremented version example
            "<p>Built with PySide6.</p>"
        )

    def closeEvent(self, event):
        print("Main Window: Closing application.")
        event.accept()
