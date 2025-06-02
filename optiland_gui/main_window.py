# optiland_gui/main_window.py
import os

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    Slot,
    Signal, 
    QTimer, 
    QSettings,  
    QByteArray,
)
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QResizeEvent
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
    QSplitter 
)

from .analysis_panel import AnalysisPanel
from .lens_editor import LensEditor
from .optiland_connector import OptilandConnector
from .optimization_panel import OptimizationPanel
from .system_properties_panel import SystemPropertiesPanel
from .viewer_panel import ViewerPanel
from .widgets.sidebar import SidebarWidget, SIDEBAR_MIN_WIDTH, COLLAPSE_THRESHOLD_WIDTH



THEME_DARK_PATH = os.path.join(os.path.dirname(__file__), "resources", "styles", "dark_theme.qss")
THEME_LIGHT_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "styles", "light_theme.qss"
)
SIDEBAR_QSS_PATH = os.path.join(os.path.dirname(__file__), "resources", "styles", "sidebar.qss")

# Define constants for QSettings
ORGANIZATION_NAME = "OptilandProject" # Or your preferred organization name
APPLICATION_NAME = "Optiland"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optiland")
        self.setGeometry(100, 100, 1600, 900)

        # Initialize QSettings for persistent layout storage
        self.settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)

        # Determine the next slot to save to (toggles between 1 and 2)
        # Default to slot 1 if no previous value is found
        self.next_save_slot_index = self.settings.value("Layouts/NextSaveSlot", 1, type=int)

        self.connector = OptilandConnector()
        self.dock_animations = {}
        self.dock_original_sizes = {}

        self.lensEditor = LensEditor(self.connector)
        self.viewerPanel = ViewerPanel(self.connector)
        self.analysisPanel = AnalysisPanel(self.connector)
        self.optimizationPanel = OptimizationPanel(self.connector)
        self.systemPropertiesPanel = SystemPropertiesPanel(self.connector)

        self.sidebar = SidebarWidget()
        
        self.placeholder_central_widget = QWidget()
        self.placeholder_central_widget.setObjectName("PlaceholderCentralWidget")

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setObjectName("MainWindowSplitter") 
        self.main_splitter.addWidget(self.sidebar)
        self.main_splitter.addWidget(self.placeholder_central_widget) 

        self.main_splitter.setHandleWidth(6)
        
        self.main_splitter.setStretchFactor(0, 0) 
        self.main_splitter.setStretchFactor(1, 1) 
        
        self.setCentralWidget(self.main_splitter)
        self.placeholder_central_widget.hide() 

        self.viewerDock = QDockWidget("System Viewer", self)
        self.viewerDock.setWidget(self.viewerPanel)
        self.viewerDock.setObjectName("ViewerDock")
        self.dock_original_sizes[self.viewerDock] = 500

        self.lensEditorDock = QDockWidget("Lens Data Editor", self)
        self.lensEditorDock.setWidget(self.lensEditor)
        self.lensEditorDock.setObjectName("LensEditorDock")
        self.dock_original_sizes[self.lensEditorDock] = 300

        self.systemPropertiesDock = QDockWidget("System Properties", self)
        self.systemPropertiesDock.setWidget(self.systemPropertiesPanel)
        self.systemPropertiesDock.setObjectName("SystemPropertiesDock")
        self.dock_original_sizes[self.systemPropertiesDock] = 300

        self.analysisPanelDock = QDockWidget("Analysis", self)
        self.analysisPanelDock.setWidget(self.analysisPanel)
        self.analysisPanelDock.setObjectName("AnalysisPanelDock")
        self.dock_original_sizes[self.analysisPanelDock] = 300

        self.optimizationPanelDock = QDockWidget("Optimization", self)
        self.optimizationPanelDock.setWidget(self.optimizationPanel)
        self.optimizationPanelDock.setObjectName("OptimizationPanelDock")
        self.dock_original_sizes[self.optimizationPanelDock] = 300

        # *** FIX: Initialize self.all_managed_docks before calling _apply_default_dock_layout ***
        self.all_managed_docks = [
            self.viewerDock, self.lensEditorDock, self.systemPropertiesDock,
            self.analysisPanelDock, self.optimizationPanelDock
        ]
        # *** END FIX ***

        self.all_managed_docks = [
            self.viewerDock, self.lensEditorDock, self.systemPropertiesDock,
            self.analysisPanelDock, self.optimizationPanelDock
        ]

        self._apply_default_dock_layout() 

        self._create_actions()
        self._create_menu_bar()
        self._create_tool_bars()

        self.current_theme_path = THEME_DARK_PATH 
        self.load_stylesheets() 

        if hasattr(self, "darkThemeAction"): 
            if self.current_theme_path == THEME_DARK_PATH:
                self.darkThemeAction.setChecked(True)
            else:
                self.lightThemeAction.setChecked(True)
        
        # Moved self.all_managed_docks initialization earlier
        # The loop for connecting dock toggle animations also needs self.all_managed_docks
        # So, it should remain after self.all_managed_docks is defined.
        for dock_widget_ref in self.all_managed_docks: # This is now correct
            action = dock_widget_ref.toggleViewAction()
            try:
                action.triggered.disconnect() 
            except (TypeError, RuntimeError): 
                pass
            action.triggered.connect(
                lambda checked, dock=dock_widget_ref: self.animate_dock_toggle(
                    dock, checked
                )
            )
        
        self._initial_narrow_check_done = False


    def _apply_default_dock_layout(self):
        """
        Applies the default dock widget layout to the right of the sidebar.
        The central widget is a QSplitter: [Sidebar | HiddenPlaceholder].
        All other docks are placed in the RightDockWidgetArea of the QMainWindow,
        arranged into a 2x2 grid.
        """
        
        # Ensure all managed docks are in a known state (visible and docked)
        for dock in self.all_managed_docks:
            if dock:
                dock.setVisible(True) 
                dock.setFloating(False)

        # --- Build the 2x2 grid in the RightDockWidgetArea ---

        # Top-Left of the grid: Lens Editor
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.lensEditorDock)
        
        # Top-Right of the grid: System Properties
        # Add it to the same area, then split horizontally with Lens Editor
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.systemPropertiesDock)
        self.splitDockWidget(self.lensEditorDock, self.systemPropertiesDock, Qt.Orientation.Horizontal)
        # Now, lensEditorDock is to the left of systemPropertiesDock in the top row of this area.

        # Bottom-Left of the grid: System Viewer
        # Add it to the RightDockWidgetArea. It will likely appear below the LDE/SP row.
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.viewerDock)
        # Split Lens Editor (which forms the top-left part) vertically with System Viewer
        self.splitDockWidget(self.lensEditorDock, self.viewerDock, Qt.Orientation.Vertical)
        # Now, viewerDock is below lensEditorDock.

        # Bottom-Right of the grid: Analysis Panel (tabbed with Optimization)
        # Add it to the RightDockWidgetArea.
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysisPanelDock)
        # Split System Properties (top-right) vertically with Analysis Panel
        self.splitDockWidget(self.systemPropertiesDock, self.analysisPanelDock, Qt.Orientation.Vertical)
        # Now, analysisPanelDock is below systemPropertiesDock.
        
        # At this point, the layout should be:
        # [LDE | SP ]
        # [VP  | AP ]
        # all within the RightDockWidgetArea.

        # Tabify Optimization Panel with the Analysis Panel
        if self.optimizationPanelDock in self.all_managed_docks:
            self.tabifyDockWidget(self.analysisPanelDock, self.optimizationPanelDock)
        
        # Ensure the primary tabs/docks are raised
        self.lensEditorDock.raise_()
        self.systemPropertiesDock.raise_()
        self.viewerDock.raise_() 
        self.analysisPanelDock.raise_() 

        # Refresh viewer panels if needed
        if self.viewerPanel.viewer2D and hasattr(self.viewerPanel.viewer2D, 'plot_optic'):
             self.viewerPanel.viewer2D.plot_optic() 
        if self.viewerPanel.viewer3D and hasattr(self.viewerPanel.viewer3D, 'render_optic'):
             self.viewerPanel.viewer3D.render_optic() 

    
    def showEvent(self, event: QResizeEvent):
        super().showEvent(event)
        if not self._initial_narrow_check_done:
            current_main_window_content_width = self.main_splitter.width() # Width of the central widget area

            target_sidebar_width = 0
            # Determine initial sidebar width and collapsed state
            # Check overall main window width, not just splitter width initially
            if self.width() < 400: # Refers to QMainWindow's total width
                target_sidebar_width = SIDEBAR_MIN_WIDTH
                self.sidebar.force_set_collapse_state(True)
            else:
                target_sidebar_width = 200 # Default expanded width for sidebar as per prompt
                self.sidebar.force_set_collapse_state(False) 
            
            # Set sizes for the main_splitter's panes: [Sidebar | HiddenPlaceholder]
            # Give the hidden placeholder a nominal small size (e.g., 0 or 1).
            # The stretch factor (0 for sidebar, 1 for placeholder) will make the placeholder
            # try to take remaining space if it were visible and if sidebar wasn't maxed out.
            # Since placeholder is hidden, sidebar effectively determines the splitter's behavior
            # for its first pane's size within its own min/max.
            # The crucial part is that the QSplitter *itself* (as the central widget)
            # is positioned correctly by QMainWindow.
            
            # Ensure the sidebar itself respects its min/max, QSplitter will handle the rest.
            # Calculate size for the (hidden) second pane of the splitter.
            # The second pane should conceptually take "the rest of the central widget space allocated to the splitter".
            # However, since it's hidden, its actual size is less relevant than the first pane's.
            # We give the sidebar its target size, and the rest to the (hidden) placeholder.
            placeholder_width = max(1, current_main_window_content_width - target_sidebar_width)
            self.main_splitter.setSizes([target_sidebar_width, placeholder_width])
            
            self._initial_narrow_check_done = True
        
        # Refresh theme choice on show
        if hasattr(self, "darkThemeAction") and hasattr(self, "lightThemeAction"): #
            is_dark = (self.current_theme_path == THEME_DARK_PATH) #
            self.darkThemeAction.setChecked(is_dark) #
            self.lightThemeAction.setChecked(not is_dark) #


    def animate_dock_toggle(self, dock_widget, show_state_after_toggle):
        animation_duration = 300 
        easing_curve = QEasingCurve.InOutQuad

        is_left_or_right_dock = self.dockWidgetArea(dock_widget) in [
            Qt.DockWidgetArea.LeftDockWidgetArea,
            Qt.DockWidgetArea.RightDockWidgetArea,
        ]
        
        default_w = 300
        default_h = 200
        
        if is_left_or_right_dock:
            original_dimension = self.dock_original_sizes.get(dock_widget, default_w) 
        else: 
            original_dimension = self.dock_original_sizes.get(dock_widget, default_h) 

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
                    animation.setEndValue(original_dimension) 
                    dock_widget.setMaximumWidth(original_dimension)
                else:
                    animation = QPropertyAnimation(dock_widget, b"maximumHeight")
                    animation.setStartValue(0)
                    animation.setEndValue(original_dimension) 
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
                    lambda: dock_widget.setMaximumWidth(original_dimension if original_dimension > 0 else 2000) 
                    if is_left_or_right_dock else 
                    dock_widget.setMaximumHeight(original_dimension if original_dimension > 0 else 2000)
                )
                animation.start(QPropertyAnimation.DeleteWhenStopped)
                self.dock_animations[dock_widget] = animation

    def _create_actions(self):
        self.newAction = QAction("&New System",self,shortcut=QKeySequence.New,triggered=self.new_system_action)
        self.openAction = QAction("&Open System...",self,shortcut=QKeySequence.Open,triggered=self.open_system_action)
        self.saveAction = QAction("&Save System",self,shortcut=QKeySequence.Save,triggered=self.save_system_action)
        self.saveAsAction = QAction("Save System &As...",self,shortcut=QKeySequence.SaveAs,triggered=self.save_system_as_action)
        self.exitAction = QAction("E&xit", self, shortcut=QKeySequence.Quit, triggered=self.close)

        # --- New Layout Actions ---
        self.loadLayout1Action = QAction("1", self, triggered=self.load_layout_1_slot)
        self.loadLayout1Action.setToolTip("Load Layout from Slot 1")

        self.loadLayout2Action = QAction("2", self, triggered=self.load_layout_2_slot)
        self.loadLayout2Action.setToolTip("Load Layout from Slot 2")

        self.saveLayoutAction = QAction("Save Current Layout", self, triggered=self.save_layout_slot)
        self.saveLayoutAction.setToolTip("Save current window layout to next available slot (1 or 2)")

        # Enable/disable load actions based on whether configurations exist
        self.loadLayout1Action.setEnabled(self.settings.contains("Layouts/Config1Geometry"))
        self.loadLayout2Action.setEnabled(self.settings.contains("Layouts/Config2Geometry"))

        self.dockAllAction = QAction("Dock All Windows",self,triggered=self.dock_all_windows_action)
        self.resetLayoutAction = QAction("Reset Window Layout",self,triggered=self.reset_windows_action)
        
        self.toggleViewerDockAction = self.viewerDock.toggleViewAction()
        self.toggleViewerDockAction.setText("Toggle System &Viewer")

        self.toggleLensEditorAction = self.lensEditorDock.toggleViewAction()
        self.toggleLensEditorAction.setText("Toggle Lens &Editor")

        self.toggleSystemPropertiesAction = self.systemPropertiesDock.toggleViewAction()
        self.toggleSystemPropertiesAction.setText("Toggle System &Properties")

        self.toggleAnalysisPanelAction = self.analysisPanelDock.toggleViewAction()
        self.toggleAnalysisPanelAction.setText("Toggle &Analysis Panel")

        self.toggleOptimizationPanelAction = self.optimizationPanelDock.toggleViewAction()
        self.toggleOptimizationPanelAction.setText("Toggle &Optimization Panel")

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
        
        viewMenu.addAction(self.toggleViewerDockAction) 
        viewMenu.addAction(self.toggleLensEditorAction)
        viewMenu.addAction(self.toggleSystemPropertiesAction)
        viewMenu.addAction(self.toggleAnalysisPanelAction)
        viewMenu.addAction(self.toggleOptimizationPanelAction)

        runMenu = menuBar.addMenu("&Run") 
        helpMenu = menuBar.addMenu("&Help")
        helpMenu.addAction(self.aboutAction)

    def _create_tool_bars(self):
        fileToolBar = self.addToolBar("File")
        fileToolBar.setObjectName("FileToolBar")
        fileToolBar.addAction(self.newAction)
        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.saveAction)
        fileToolBar.addSeparator()
        # --- Add new layout buttons before Dock All Windows ---
        fileToolBar.addSeparator()
        fileToolBar.addAction(self.loadLayout1Action)
        fileToolBar.addAction(self.loadLayout2Action)
        fileToolBar.addAction(self.saveLayoutAction)
        fileToolBar.addSeparator()
        # --- End of new layout buttons ---
        fileToolBar.addAction(self.dockAllAction) 
        fileToolBar.addAction(self.resetLayoutAction) 

    def load_stylesheets(self):
        style_str = ""
        try:
            with open(self.current_theme_path) as f_theme:
                style_str += f_theme.read()
            print(f"Main theme loaded: {self.current_theme_path}")
        except FileNotFoundError:
            print(f"Main theme stylesheet not found: {self.current_theme_path}")
            QMessageBox.warning(self, "Theme Error", f"Stylesheet not found: {os.path.basename(self.current_theme_path)}")
        except Exception as e:
            print(f"Error loading main theme stylesheet: {e}")
            QMessageBox.critical(self, "Theme Error", f"Error loading main stylesheet: {e}")
            
        if os.path.exists(SIDEBAR_QSS_PATH):
            try:
                with open(SIDEBAR_QSS_PATH) as f_sidebar:
                    style_str += "\n" + f_sidebar.read()
                print(f"Sidebar stylesheet loaded: {SIDEBAR_QSS_PATH}")
            except Exception as e:
                print(f"Error loading sidebar stylesheet: {e}")
        else:
            print(f"Sidebar stylesheet not found: {SIDEBAR_QSS_PATH}")

        self.setStyleSheet("") 
        self.setStyleSheet(style_str)
        
        if hasattr(self, "darkThemeAction") and hasattr(self, "lightThemeAction"):
            is_dark = (self.current_theme_path == THEME_DARK_PATH)
            self.darkThemeAction.setChecked(is_dark)
            self.lightThemeAction.setChecked(not is_dark)


    @Slot(str)
    def switch_theme(self, theme_path):
        if theme_path != self.current_theme_path:
            self.current_theme_path = theme_path
            self.load_stylesheets() 

    @Slot()
    def new_system_action(self):
        self.connector.new_system()
        print("Main Window: New System action triggered")

    @Slot()
    def open_system_action(self):
        filepath, _ = QFileDialog.getOpenFileName(self,"Open Optiland System","", "Optiland JSON Files (*.json);;All Files (*)")
        if filepath:
            self.connector.load_optic_from_file(filepath)
            print(f"Main Window: Open System action triggered - {filepath}")

    @Slot()
    def save_system_action(self):
        current_path = self.connector.get_current_filepath()
        if current_path:
            self.connector.save_optic_to_file(current_path)
            print(f"Main Window: Save System action triggered - {current_path}")
        else:
            self.save_system_as_action()

    @Slot()
    def save_system_as_action(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Optiland System As...", "", "Optiland JSON Files (*.json);;All Files (*)")
        if filepath:
            if not filepath.lower().endswith(".json") and "(*.json)" in _.split(";;")[0]: 
                filepath += ".json"
            self.connector.save_optic_to_file(filepath)
            print(f"Main Window: Save System As action triggered - {filepath}")

    @Slot()
    def about_action(self):
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("About Optiland GUI")
        layout = QVBoxLayout(about_dialog)
        about_text = QLabel(
            "<p><b>Optiland GUI</b></p>"
            "<p>A modern interface for the Optiland optical simulation package.</p>"
            "<p>Version: 0.1.8 (Sidebar Implemented)</p>" 
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
        about_dialog.exec()


    @Slot()
    def dock_all_windows_action(self):
        for dock in self.all_managed_docks:
            if dock:
                dock.setFloating(False)
                if not dock.isVisible(): 
                    if dock.toggleViewAction():
                        dock.toggleViewAction().trigger() 
                else: 
                    dock.show() 
                    dock.raise_()
                if dock.toggleViewAction():
                    dock.toggleViewAction().setChecked(True)
        print("Main Window: Dock All Windows action triggered")

    @Slot()
    def reset_windows_action(self):
        self._apply_default_dock_layout()
        for dock in self.all_managed_docks:
            if dock and dock.toggleViewAction():
                dock.toggleViewAction().setChecked(not dock.isHidden())
        print("Main Window: Reset Windows action triggered")

    @Slot()
    def save_layout_slot(self):
        """Saves the current window geometry and dock/toolbar state to the next available slot."""
        target_slot = self.next_save_slot_index
        
        window_geometry = self.saveGeometry()
        dock_toolbar_state = self.saveState()

        self.settings.setValue(f"Layouts/Config{target_slot}Geometry", window_geometry)
        self.settings.setValue(f"Layouts/Config{target_slot}State", dock_toolbar_state)
        
        QMessageBox.information(self, "Layout Saved", 
                                f"The current window layout was saved to configuration - {target_slot}")
        
        # Toggle to the other slot for the next save
        self.next_save_slot_index = 2 if target_slot == 1 else 1
        self.settings.setValue("Layouts/NextSaveSlot", self.next_save_slot_index)

        # Update enabled state of load buttons
        self.loadLayout1Action.setEnabled(self.settings.contains("Layouts/Config1Geometry"))
        self.loadLayout2Action.setEnabled(self.settings.contains("Layouts/Config2Geometry"))
        print(f"Layout saved to slot {target_slot}. Next save will be to slot {self.next_save_slot_index}.")

    def _load_layout_from_slot(self, slot_number):
        """Helper function to load layout from a given slot number."""
        geometry_key = f"Layouts/Config{slot_number}Geometry"
        state_key = f"Layouts/Config{slot_number}State"

        if self.settings.contains(geometry_key) and self.settings.contains(state_key):
            window_geometry = self.settings.value(geometry_key)
            dock_toolbar_state = self.settings.value(state_key)

            if isinstance(window_geometry, QByteArray) and isinstance(dock_toolbar_state, QByteArray):
                if not self.restoreGeometry(window_geometry):
                    print(f"Warning: Failed to restore window geometry from slot {slot_number}.")
                if not self.restoreState(dock_toolbar_state):
                    print(f"Warning: Failed to restore dock/toolbar state from slot {slot_number}.")
                
                # After restoring state, docks might need to be shown explicitly if saveState doesn't cover visibility fully for complex cases
                # self._apply_default_dock_layout() # Or a simpler show all docks.
                # The default _apply_default_dock_layout might override the restored state too much.
                # Often restoreState is enough. We can refine this if docks don't appear correctly.
                for dock in self.all_managed_docks: # Ensure they are visible after restore
                    if dock: # Check if dock object exists
                         # Check if the restored state made it visible, if not, show.
                         # This might conflict if restoreState is perfect.
                         # A simple dock.show() might be too aggressive if some were meant to be hidden.
                         # For now, let's assume restoreState handles visibility.
                         pass


                QMessageBox.information(self, "Layout Loaded", 
                                        f"Layout from configuration - {slot_number} has been loaded.")
            else:
                QMessageBox.warning(self, "Load Error", 
                                    f"Invalid layout data found in configuration - {slot_number}.")
        else:
            QMessageBox.information(self, "Load Layout", 
                                    f"No layout saved in configuration - {slot_number}.")

    @Slot()
    def load_layout_1_slot(self):
        """Loads the layout from configuration slot 1."""
        print("Loading layout from slot 1...")
        self._load_layout_from_slot(1)

    @Slot()
    def load_layout_2_slot(self):
        """Loads the layout from configuration slot 2."""
        print("Loading layout from slot 2...")
        self._load_layout_from_slot(2)

    def closeEvent(self, event):
        print("Main Window: Closing application.")
        event.accept()