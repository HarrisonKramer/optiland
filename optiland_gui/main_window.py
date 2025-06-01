# optiland_gui/main_window.py
import os

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    Slot,
    Signal
)  
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget, 
)

from .analysis_panel import AnalysisPanel
from .lens_editor import LensEditor
from .optiland_connector import OptilandConnector
from .optimization_panel import OptimizationPanel
from .system_properties_panel import SystemPropertiesPanel
from .viewer_panel import ViewerPanel

THEME_DARK_PATH = os.path.join(os.path.dirname(__file__), "resources", "dark_theme.qss")
THEME_LIGHT_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "light_theme.qss"
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optiland GUI")
        self.setGeometry(100, 100, 1600, 900)

        self.connector = OptilandConnector()
        self.dock_animations = {}  # To store animations
        self.dock_original_sizes = {}  # To store original sizes for restore

        # --- Create Panels ---
        self.lensEditor = LensEditor(self.connector)
        self.viewerPanel = ViewerPanel(self.connector)
        self.analysisPanel = AnalysisPanel(self.connector)
        self.optimizationPanel = OptimizationPanel(self.connector)
        self.systemPropertiesPanel = SystemPropertiesPanel(self.connector)


        placeholder_central_widget = QWidget() 
        placeholder_central_widget.setStyleSheet("background-color: transparent;") 
        self.setCentralWidget(placeholder_central_widget)
        self.centralWidget().hide() 

        # --- Dock Widgets ---
        
        # System Viewer Dock
        self.viewerDock = QDockWidget("System Viewer", self)
        self.viewerDock.setWidget(self.viewerPanel)
        self.viewerDock.setObjectName("ViewerDock")
        self.dock_original_sizes[self.viewerDock] = 500
        
        # Lens Editor Dock
        self.lensEditorDock = QDockWidget("Lens Data Editor", self)
        self.lensEditorDock.setWidget(self.lensEditor)
        self.lensEditorDock.setObjectName("LensEditorDock")
        self.dock_original_sizes[self.lensEditorDock] = 300

        # System Properties Dock
        self.systemPropertiesDock = QDockWidget("System Properties", self)
        self.systemPropertiesDock.setWidget(self.systemPropertiesPanel)
        self.tabifyDockWidget(self.lensEditorDock, self.systemPropertiesDock)
        self.lensEditorDock.raise_()
        self.systemPropertiesDock.setObjectName("SystemPropertiesDock")
        self.dock_original_sizes[self.systemPropertiesDock] = 300

        # Analysis and Optimization Docks
        self.analysisPanelDock = QDockWidget("Analysis", self)
        self.analysisPanelDock.setWidget(self.analysisPanel)
        self.analysisPanelDock.setObjectName("AnalysisPanelDock")
        self.dock_original_sizes[self.analysisPanelDock] = 300

        self.optimizationPanelDock = QDockWidget("Optimization", self)
        self.optimizationPanelDock.setWidget(self.optimizationPanel)
        self.optimizationPanelDock.setObjectName("OptimizationPanelDock")
        self.dock_original_sizes[self.optimizationPanelDock] = 300
        
        self._apply_default_dock_layout() 

        self._create_actions()
        self._create_menu_bar()
        self._create_tool_bars()

        self.current_theme_path = THEME_DARK_PATH
        if hasattr(self, "darkThemeAction"):
            if self.current_theme_path == THEME_DARK_PATH:
                self.darkThemeAction.setChecked(True)
            else:
                self.lightThemeAction.setChecked(True)
        
        self.load_stylesheet(self.current_theme_path)

        # Connect toggle actions for animation
        self.all_managed_docks = [ # Keep a list for easier management
            self.viewerDock, self.lensEditorDock, self.systemPropertiesDock,
            self.analysisPanelDock, self.optimizationPanelDock
        ]

        for dock_widget_ref in self.all_managed_docks:
            action = dock_widget_ref.toggleViewAction()
            try:
                # Disconnect all previous connections to this specific signal instance
                # This is tricky if multiple slots were connected.
                # A common pattern if replacing is to ensure action is fresh or manage connections carefully.
                # For simplicity here, we'll assume previous attempts to disconnect were sufficient
                # or that connecting our specific lambda is the primary goal.
                # If multiple animations run, that's a bug to fix in connection logic.
                # Let's clear specific connections if possible or rely on new connection.
                pass # Assuming disconnect logic will be handled if necessary by Qt or by careful single connections
            except RuntimeError:
                pass 
            
            action.triggered.connect(
                lambda checked, dock=dock_widget_ref: self.animate_dock_toggle(
                    dock, checked
                )
            )

    def _apply_default_dock_layout(self):
        """Helper function to set up the initial/reset dock layout."""
        # Ensure all docks are initially visible and not floating for reset
        # Order of addDockWidget and tabifyDockWidget matters for initial layout.

        # Left Docks: Lens Editor and System Properties, tabbed
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.lensEditorDock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.systemPropertiesDock)
        self.tabifyDockWidget(self.lensEditorDock, self.systemPropertiesDock)
        self.lensEditorDock.raise_()
        lens_editor_parent = self.lensEditorDock.parentWidget()
        if lens_editor_parent and isinstance(lens_editor_parent, QWidget): # Check if parent is a QWidget
            lens_editor_parent.resize(
                self.dock_original_sizes.get(self.lensEditorDock, 350), 
                lens_editor_parent.height()
            )

        # Viewer Dock: Initially on the right, taking significant space
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.viewerDock)
        self.viewerDock.raise_()
        self.viewerDock.resize( # Resize the dock itself
            self.dock_original_sizes.get(self.viewerDock, 700),
            self.viewerDock.sizeHint().height() # Use size hint for height or a fixed value
        )


        # Analysis and Optimization Docks: Tabbed, also on the right, below/beside viewer
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysisPanelDock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.optimizationPanelDock)
        self.tabifyDockWidget(self.analysisPanelDock, self.optimizationPanelDock)
        self.analysisPanelDock.raise_()
        analysis_parent = self.analysisPanelDock.parentWidget()
        if analysis_parent and isinstance(analysis_parent, QWidget):
            analysis_parent.resize(
                self.dock_original_sizes.get(self.analysisPanelDock, 300),
                analysis_parent.height()
            )
        
        # Ensure all docks are visible after setting layout
        for dock in [self.lensEditorDock, self.systemPropertiesDock, self.viewerDock, self.analysisPanelDock, self.optimizationPanelDock]:
            if dock: # Check if dock is initialized
                dock.setVisible(True)
                dock.setFloating(False)


    def animate_dock_toggle(self, dock_widget, show_state_after_toggle):
        # ... (animation logic from previous correct version) ...
        # This logic should be largely okay, ensure original_width/height are handled for viewerDock if needed.
        # For a dock like viewerDock (Right area), original_width is primary from dock_original_sizes.
        animation_duration = 300 # ms
        easing_curve = QEasingCurve.InOutQuad

        is_left_or_right_dock = self.dockWidgetArea(dock_widget) in [
            Qt.DockWidgetArea.LeftDockWidgetArea,
            Qt.DockWidgetArea.RightDockWidgetArea,
        ]
        
        default_w = 300
        default_h = 200
        
        if is_left_or_right_dock:
            original_dimension = self.dock_original_sizes.get(dock_widget, default_w) # This is width
        else: # Top or Bottom dock
            # Assuming dock_original_sizes might store height for T/B docks, or use a default
            original_dimension = self.dock_original_sizes.get(dock_widget, default_h) # This should be height

        # Stop any existing animation for this dock
        if (
            dock_widget in self.dock_animations
            and self.dock_animations[dock_widget].state() == QPropertyAnimation.Running
        ):
            self.dock_animations[dock_widget].stop()

        current_visibility = not dock_widget.isHidden()
        
        if show_state_after_toggle:
            if not current_visibility:
                dock_widget.show()
                dock_widget.raise_()
                
                if is_left_or_right_dock:
                    animation = QPropertyAnimation(dock_widget, b"maximumWidth")
                    animation.setStartValue(0)
                    animation.setEndValue(original_dimension) # original_dimension is width
                    dock_widget.setMaximumWidth(original_dimension)
                else:
                    animation = QPropertyAnimation(dock_widget, b"maximumHeight")
                    animation.setStartValue(0)
                    animation.setEndValue(original_dimension) # original_dimension is height
                    dock_widget.setMaximumHeight(original_dimension)
                
                animation.setDuration(animation_duration)
                animation.setEasingCurve(easing_curve)
                animation.finished.connect(lambda: dock_widget.resize(original_dimension, dock_widget.height()) if is_left_or_right_dock else dock_widget.resize(dock_widget.width(), original_dimension))
                animation.start(QPropertyAnimation.DeleteWhenStopped)
                self.dock_animations[dock_widget] = animation
            else:
                dock_widget.raise_()
                if is_left_or_right_dock:
                    dock_widget.setMaximumWidth(original_dimension)
                else:
                    dock_widget.setMaximumHeight(original_dimension)
        else:
            if current_visibility:
                current_size = dock_widget.width() if is_left_or_right_dock else dock_widget.height()
                if is_left_or_right_dock:
                    animation = QPropertyAnimation(dock_widget, b"maximumWidth")
                    animation.setStartValue(current_size)
                    animation.setEndValue(0)
                else:
                    animation = QPropertyAnimation(dock_widget, b"maximumHeight")
                    animation.setStartValue(current_size)
                    animation.setEndValue(0)

                animation.setDuration(animation_duration)
                animation.setEasingCurve(easing_curve)
                animation.finished.connect(dock_widget.hide)
                animation.finished.connect(
                    lambda: dock_widget.setMaximumWidth(original_dimension) if is_left_or_right_dock else dock_widget.setMaximumHeight(2000)
                )
                animation.start(QPropertyAnimation.DeleteWhenStopped)
                self.dock_animations[dock_widget] = animation


    def _create_actions(self):
        self.newAction = QAction("&New System",self,shortcut=QKeySequence.New,triggered=self.new_system_action)
        self.openAction = QAction("&Open System...",self,shortcut=QKeySequence.Open,triggered=self.open_system_action)
        self.saveAction = QAction("&Save System",self,shortcut=QKeySequence.Save,triggered=self.save_system_action)
        self.saveAsAction = QAction("Save System &As...",self,shortcut=QKeySequence.SaveAs,triggered=self.save_system_as_action)
        self.exitAction = QAction("E&xit", self, shortcut=QKeySequence.Quit, triggered=self.close)

        self.dockAllAction = QAction("Dock All Windows",self,triggered=self.dock_all_windows_action)
        self.resetLayoutAction = QAction("Reset Window Layout",self,triggered=self.reset_windows_action)
        
        # View actions for toggling docks
        self.toggleViewerDockAction = self.viewerDock.toggleViewAction()
        self.toggleViewerDockAction.setText("Toggle System &Viewer")
        self.toggleViewerDockAction.setCheckable(True)

        self.toggleLensEditorAction = self.lensEditorDock.toggleViewAction()
        self.toggleLensEditorAction.setText("Toggle Lens &Editor")
        self.toggleLensEditorAction.setCheckable(True)

        self.toggleSystemPropertiesAction = self.systemPropertiesDock.toggleViewAction()
        self.toggleSystemPropertiesAction.setText("Toggle System &Properties")
        self.toggleSystemPropertiesAction.setCheckable(True)

        self.toggleAnalysisPanelAction = self.analysisPanelDock.toggleViewAction()
        self.toggleAnalysisPanelAction.setText("Toggle &Analysis Panel")
        self.toggleAnalysisPanelAction.setCheckable(True)

        self.toggleOptimizationPanelAction = self.optimizationPanelDock.toggleViewAction()
        self.toggleOptimizationPanelAction.setText("Toggle &Optimization Panel")
        self.toggleOptimizationPanelAction.setCheckable(True)

        self.themeActionGroup = QActionGroup(self)
        self.themeActionGroup.setExclusive(True)
        self.darkThemeAction = QAction("Dark Theme", self, checkable=True)
        self.darkThemeAction.triggered.connect(lambda: self.switch_theme(THEME_DARK_PATH))
        self.themeActionGroup.addAction(self.darkThemeAction)
        self.lightThemeAction = QAction("Light Theme", self, checkable=True)
        self.lightThemeAction.triggered.connect(lambda: self.switch_theme(THEME_LIGHT_PATH))
        self.themeActionGroup.addAction(self.lightThemeAction)

        self.aboutAction = QAction("&About Optiland GUI", self, triggered=self.about_action)
        self.undoAction = QAction("&Undo",self,shortcut=QKeySequence.Undo,triggered=self.connector.undo)
        self.undoAction.setEnabled(False)
        self.redoAction = QAction("&Redo",self,shortcut=QKeySequence.Redo,triggered=self.connector.redo)
        self.redoAction.setEnabled(False)
        self.connector.undoStackAvailabilityChanged.connect(self.undoAction.setEnabled)
        self.connector.redoStackAvailabilityChanged.connect(self.redoAction.setEnabled)


    def _create_menu_bar(self):
        menuBar = self.menuBar()
        # ... (File, Edit menus as before) ...
        fileMenu = menuBar.addMenu("&File")
        fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.saveAction)
        fileMenu.addAction(self.saveAsAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)

        editMenu = menuBar.addMenu("&Edit")
        editMenu.addAction(self.undoAction)
        editMenu.addAction(self.redoAction)

        viewMenu = menuBar.addMenu("&View")
        viewMenu.addAction(self.dockAllAction)
        viewMenu.addAction(self.resetLayoutAction)
        viewMenu.addSeparator()
        themeMenu = viewMenu.addMenu("&Theme")
        themeMenu.addAction(self.darkThemeAction)
        themeMenu.addAction(self.lightThemeAction)
        viewMenu.addSeparator()
        
        viewMenu.addAction(self.toggleViewerDockAction) # ADDED
        viewMenu.addAction(self.toggleLensEditorAction)
        viewMenu.addAction(self.toggleSystemPropertiesAction)
        viewMenu.addAction(self.toggleAnalysisPanelAction)
        viewMenu.addAction(self.toggleOptimizationPanelAction)

        runMenu = menuBar.addMenu("&Run")
        helpMenu = menuBar.addMenu("&Help")
        helpMenu.addAction(self.aboutAction)


    def _create_tool_bars(self):
        # ... (File toolbar as before, with dockAll and resetLayout actions) ...
        fileToolBar = self.addToolBar("File")
        fileToolBar.setObjectName("FileToolBar")
        fileToolBar.addAction(self.newAction)
        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.saveAction)
        fileToolBar.addSeparator()
        fileToolBar.addAction(self.dockAllAction)
        fileToolBar.addAction(self.resetLayoutAction)

    def load_stylesheet(self, filepath):
        # ... (as before) ...
        try:
            with open(filepath) as f:
                style_str = f.read()
                self.setStyleSheet("")
                self.setStyleSheet(style_str)
                self.current_theme_path = filepath
                if hasattr(self, "darkThemeAction") and hasattr(self, "lightThemeAction"):
                    is_dark = (filepath == THEME_DARK_PATH)
                    self.darkThemeAction.setChecked(is_dark)
                    self.lightThemeAction.setChecked(not is_dark)
                print(f"Stylesheet loaded: {filepath}")
        except FileNotFoundError:
            print(f"Stylesheet not found: {filepath}")
            QMessageBox.warning(self, "Theme Error", f"Stylesheet not found: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Error loading stylesheet: {e}")
            QMessageBox.critical(self, "Theme Error", f"Error loading stylesheet: {e}")


    @Slot(str)
    def switch_theme(self, theme_path):
        # ... (as before) ...
        if theme_path != self.current_theme_path:
            self.load_stylesheet(theme_path)

    @Slot()
    def new_system_action(self): # ... (as before) ...
        self.connector.new_system()
        print("Main Window: New System action triggered")

    @Slot()
    def open_system_action(self): # ... (as before) ...
        filepath, _ = QFileDialog.getOpenFileName(self,"Open Optiland System","", "Optiland JSON Files (*.json);;All Files (*)")
        if filepath:
            self.connector.load_optic_from_file(filepath)
            print(f"Main Window: Open System action triggered - {filepath}")

    @Slot()
    def save_system_action(self): # ... (as before) ...
        current_path = self.connector.get_current_filepath()
        if current_path:
            self.connector.save_optic_to_file(current_path)
            print(f"Main Window: Save System action triggered - {current_path}")
        else:
            self.save_system_as_action()

    @Slot()
    def save_system_as_action(self): # ... (as before) ...
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Optiland System As...", "", "Optiland JSON Files (*.json);;All Files (*)")
        if filepath:
            if not filepath.lower().endswith(".json") and "(*.json)" in _.split(";;")[0]:
                filepath += ".json"
            self.connector.save_optic_to_file(filepath)
            print(f"Main Window: Save System As action triggered - {filepath}")

    @Slot()
    def about_action(self): # ... (as before, maybe update version string) ...
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("About Optiland GUI")
        layout = QVBoxLayout(about_dialog)
        about_text = QLabel(
            "<p><b>Optiland GUI</b></p>"
            "<p>A modern interface for the Optiland optical simulation package.</p>"
            "<p>Version: 0.1.7 (Viewer Dock & System Props Resize)</p>"
            "<p>Built with PySide6.</p>"
        )
        about_text.setTextFormat(Qt.TextFormat.RichText)
        about_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(about_text)
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(about_dialog.accept)
        layout.addWidget(ok_button, alignment=Qt.AlignmentFlag.AlignCenter)
        about_dialog.setLayout(layout)
        about_dialog.setMinimumSize(350, 220)
        about_dialog.setWindowOpacity(0.0)
        self.about_dialog_animation = QPropertyAnimation(about_dialog, b"windowOpacity")
        self.about_dialog_animation.setDuration(300)
        self.about_dialog_animation.setStartValue(0.0)
        self.about_dialog_animation.setEndValue(1.0)
        self.about_dialog_animation.setEasingCurve(QEasingCurve.InOutQuad)
        about_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.about_dialog_animation.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        about_dialog.exec_()


    @Slot()
    def dock_all_windows_action(self):
        for dock in self.all_managed_docks: # Use the list of all docks
            if dock: # Check if dock object exists
                dock.setFloating(False)
                dock.show()
                dock.raise_()
                toggle_action = dock.toggleViewAction()
                if toggle_action:
                    toggle_action.setChecked(True)
        print("Main Window: Dock All Windows action triggered")

    @Slot()
    def reset_windows_action(self):
        self._apply_default_dock_layout()
        for dock in self.all_managed_docks: # Use the list of all docks
            if dock: # Check if dock object exists
                toggle_action = dock.toggleViewAction()
                if toggle_action:
                    toggle_action.setChecked(not dock.isHidden())
        print("Main Window: Reset Windows action triggered")


    def closeEvent(self, event): # ... (as before) ...
        print("Main Window: Closing application.")
        event.accept()