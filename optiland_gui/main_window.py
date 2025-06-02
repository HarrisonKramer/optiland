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
    QEvent
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
    QSplitter, 
    QMenuBar,
    QToolBar
)

from .analysis_panel import AnalysisPanel
from .lens_editor import LensEditor
from .optiland_connector import OptilandConnector
from .optimization_panel import OptimizationPanel
from .system_properties_panel import SystemPropertiesPanel
from .viewer_panel import ViewerPanel
from .widgets.sidebar import SidebarWidget, SIDEBAR_MIN_WIDTH, SIDEBAR_MAX_WIDTH, COLLAPSE_THRESHOLD_WIDTH 
from .widgets.custom_title_bar import CustomTitleBar

try:
    from .resources import resources_rc
except ImportError as e:
    print(f"Warning (main_window.py): Could not import resources_rc.py: {e}")

THEME_DARK_PATH = os.path.join(os.path.dirname(__file__), "resources", "styles", "dark_theme.qss")
THEME_LIGHT_PATH = os.path.join(os.path.dirname(__file__), "resources", "styles", "light_theme.qss")
SIDEBAR_QSS_PATH = os.path.join(os.path.dirname(__file__), "resources", "styles", "sidebar.qss")
CUSTOM_TITLE_BAR_QSS_PATH = os.path.join(os.path.dirname(__file__), "resources", "styles", "custom_title_bar.qss")

ORGANIZATION_NAME = "OptilandProject"
APPLICATION_NAME = "OptilandGUI"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setWindowTitle("Optiland GUI")
        self.setGeometry(100, 100, 1600, 900)

        self.settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)
        self.next_save_slot_index = self.settings.value("Layouts/NextSaveSlot", 1, type=int)
        self.connector = OptilandConnector()
        self.dock_animations = {}
        self.dock_original_sizes = {} 

        # --- Create Panels (content for docks) ---
        self.lensEditor = LensEditor(self.connector)
        self.viewerPanel = ViewerPanel(self.connector)
        self.analysisPanel = AnalysisPanel(self.connector)
        self.optimizationPanel = OptimizationPanel(self.connector)
        self.systemPropertiesPanel = SystemPropertiesPanel(self.connector)

        # --- Create QDockWidget wrappers (including one for the sidebar) ---
        self._setup_all_dock_widgets() 

        # --- Main Menu Bar Instance (for CustomTitleBar) ---
        self._actual_menu_bar_instance = QMenuBar(self) # This QMenuBar will be put into the CustomTitleBar
        self._create_actions() 
        self._populate_main_menu_bar(self._actual_menu_bar_instance)

        # --- Custom Title Bar (as a QWidget) ---
        self.custom_title_bar_widget = CustomTitleBar(self._actual_menu_bar_instance, self)
        self.custom_title_bar_widget.minimize_requested.connect(self.showMinimized)
        self.custom_title_bar_widget.maximize_restore_requested.connect(self._handle_maximize_restore)
        self.custom_title_bar_widget.close_requested.connect(self.close)
        self.connector.opticLoaded.connect(self._update_project_name_in_title_bar)
        self.connector.opticChanged.connect(self._update_project_name_in_title_bar)

        # --- Create a QToolBar to host the CustomTitleBar widget ---
        self.title_bar_as_toolbar = QToolBar("CustomTitleBarToolbar")
        self.title_bar_as_toolbar.setObjectName("CustomTitleBarToolbar")
        self.title_bar_as_toolbar.setMovable(False) # Prevent user from moving it
        self.title_bar_as_toolbar.setFloatable(False)
        self.title_bar_as_toolbar.addWidget(self.custom_title_bar_widget) # Add CustomTitleBar QWidget here
        self.addToolBar(Qt.TopToolBarArea, self.title_bar_as_toolbar)

        # --- "Quick Actions" Toolbar (as a standard QToolBar) ---
        self.quick_actions_toolbar = QToolBar("QuickActionsToolbar")
        self.quick_actions_toolbar.setObjectName("QuickActionsToolbar")
        self.quick_actions_toolbar.setMovable(True) # Allow moving if desired, or set to False
        self._populate_quick_actions_toolbar(self.quick_actions_toolbar)
        self.addToolBarBreak(Qt.TopToolBarArea) # Force quick_actions_toolbar to a new line below title_bar_as_toolbar
        self.addToolBar(Qt.TopToolBarArea, self.quick_actions_toolbar)
        
        # --- Central Widget (Placeholder for Docking Area) ---
        # QMainWindow arranges docks around this central widget.
        self.main_docking_area_placeholder = QWidget()
        self.main_docking_area_placeholder.setObjectName("MainDockingAreaPlaceholder")
        # Optional: give it a minimum size or expanding policy if needed, but QMainWindow manages this area.
        # self.main_docking_area_placeholder.setStyleSheet("background-color: #444;") # For debugging visibility
        self.setCentralWidget(self.main_docking_area_placeholder)

        # --- Apply Dock Layout ---
        # Docks are children of QMainWindow and will arrange around the central widget,
        # below the toolbars.
        self.setDockNestingEnabled(True) # Important for complex dock layouts
        self._apply_revised_default_dock_layout()

        # --- Load Stylesheets & Final Setup ---
        self.current_theme_path = THEME_DARK_PATH
        self.load_stylesheets()
        if hasattr(self, "darkThemeAction"):
            self.darkThemeAction.setChecked(self.current_theme_path == THEME_DARK_PATH)
            self.lightThemeAction.setChecked(self.current_theme_path != THEME_DARK_PATH)
        
        self._connect_dock_animations()
        self._initial_narrow_check_done = False
        self._update_project_name_in_title_bar() # Initial update for project name

    def _setup_all_dock_widgets(self):
        """Initializes all QDockWidget instances, including the sidebar as a dock."""
        # Sidebar Dock
        self.sidebar_content_widget = SidebarWidget(self) 
        self.sidebarDock = QDockWidget("NavigationSidebar", self) 
        self.sidebarDock.setWidget(self.sidebar_content_widget)
        self.sidebarDock.setObjectName("SidebarDockWidget")
        self.sidebarDock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.sidebarDock.setTitleBarWidget(QWidget()) 
        self.sidebarDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Other Docks
        self.viewerDock = QDockWidget("System Viewer", self)
        self.viewerDock.setWidget(self.viewerPanel)
        self.viewerDock.setObjectName("ViewerDock")
        self.viewerDock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        self.lensEditorDock = QDockWidget("Lens Data Editor", self)
        self.lensEditorDock.setWidget(self.lensEditor)
        self.lensEditorDock.setObjectName("LensEditorDock")
        self.lensEditorDock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        self.systemPropertiesDock = QDockWidget("System Properties", self)
        self.systemPropertiesDock.setWidget(self.systemPropertiesPanel)
        self.systemPropertiesDock.setObjectName("SystemPropertiesDock")
        self.systemPropertiesDock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        
        self.analysisPanelDock = QDockWidget("Analysis", self)
        self.analysisPanelDock.setWidget(self.analysisPanel)
        self.analysisPanelDock.setObjectName("AnalysisPanelDock")
        self.analysisPanelDock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        self.optimizationPanelDock = QDockWidget("Optimization", self)
        self.optimizationPanelDock.setWidget(self.optimizationPanel)
        self.optimizationPanelDock.setObjectName("OptimizationPanelDock")
        self.optimizationPanelDock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        self.all_managed_docks = [
            self.sidebarDock, self.viewerDock, self.lensEditorDock, 
            self.systemPropertiesDock, self.analysisPanelDock, self.optimizationPanelDock
        ]

    def _apply_revised_default_dock_layout(self):
        """
        Applies the default dock layout: SidebarDock on the left, 
        and the 2x2 grid of other docks filling the remaining space.
        """
        for dock in self.all_managed_docks:
            if dock:
                dock.setFloating(False)
                dock.show() 

        self.addDockWidget(Qt.LeftDockWidgetArea, self.sidebarDock)
        
        # These docks will now be correctly placed relative to the central widget,
        # and below the toolbars.
        self.addDockWidget(Qt.RightDockWidgetArea, self.lensEditorDock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.systemPropertiesDock)
        self.splitDockWidget(self.lensEditorDock, self.systemPropertiesDock, Qt.Horizontal)

        self.addDockWidget(Qt.RightDockWidgetArea, self.viewerDock)
        self.splitDockWidget(self.lensEditorDock, self.viewerDock, Qt.Vertical)
        
        self.addDockWidget(Qt.RightDockWidgetArea, self.analysisPanelDock)
        self.splitDockWidget(self.systemPropertiesDock, self.analysisPanelDock, Qt.Vertical)
        
        self.tabifyDockWidget(self.analysisPanelDock, self.optimizationPanelDock)
        
        self.sidebarDock.raise_()
        self.lensEditorDock.raise_()
        self.systemPropertiesDock.raise_()
        self.viewerDock.raise_() 
        self.analysisPanelDock.raise_() 

        if self.viewerPanel.viewer2D and hasattr(self.viewerPanel.viewer2D, 'plot_optic'):
             self.viewerPanel.viewer2D.plot_optic() 
        if self.viewerPanel.viewer3D and hasattr(self.viewerPanel.viewer3D, 'render_optic'):
             self.viewerPanel.viewer3D.render_optic()

    def showEvent(self, event: QResizeEvent):
        super().showEvent(event)
        if not self._initial_narrow_check_done:
            if hasattr(self, 'sidebar_content_widget') and self.sidebar_content_widget:
                if self.width() < (SIDEBAR_MAX_WIDTH + 300): 
                    self.sidebar_content_widget.force_set_collapse_state(True)
                    if self.sidebarDock.width() > SIDEBAR_MIN_WIDTH: # Check dock width
                         self.resizeDocks([self.sidebarDock], [SIDEBAR_MIN_WIDTH], Qt.Horizontal)
                else:
                    self.sidebar_content_widget.force_set_collapse_state(False)
                    if self.sidebarDock.width() < 150 : 
                         self.resizeDocks([self.sidebarDock], [150], Qt.Horizontal)
            self._initial_narrow_check_done = True
        
        if hasattr(self, "darkThemeAction") and hasattr(self, "lightThemeAction"):
            is_dark = (self.current_theme_path == THEME_DARK_PATH)
            self.darkThemeAction.setChecked(is_dark)
            self.lightThemeAction.setChecked(not is_dark)
            
    def _connect_dock_animations(self):
        """Connects toggle actions for dock animations."""
        if not hasattr(self, 'all_managed_docks'): 
            return
        for dock_widget_ref in self.all_managed_docks:
            if dock_widget_ref: 
                action = dock_widget_ref.toggleViewAction()
                if action: 
                    try:
                        action.triggered.disconnect() 
                    except (TypeError, RuntimeError): 
                        pass
                    action.triggered.connect(
                        lambda checked, dock=dock_widget_ref: self.animate_dock_toggle(
                            dock, checked
                        )
                    )

    def _populate_main_menu_bar(self, menu_bar: QMenuBar):
        """Populates the passed QMenuBar instance with application menus and actions."""
        fileMenu = menu_bar.addMenu("&File")
        fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.saveAction)
        fileMenu.addAction(self.saveAsAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)

        editMenu = menu_bar.addMenu("&Edit")
        editMenu.addAction(self.undoAction)
        editMenu.addAction(self.redoAction)

        viewMenu = menu_bar.addMenu("&View")
        viewMenu.addAction(self.dockAllAction)
        viewMenu.addAction(self.resetLayoutAction)
        viewMenu.addSeparator()
        themeMenu = viewMenu.addMenu("&Theme")
        themeMenu.addAction(self.darkThemeAction)
        themeMenu.addAction(self.lightThemeAction)
        viewMenu.addSeparator()
        
        if hasattr(self, 'sidebarDock') and self.sidebarDock: 
            action = self.sidebarDock.toggleViewAction()
            action.setText("Toggle Navigation Sidebar") # Ensure text is set
            viewMenu.addAction(action)
        if hasattr(self, 'viewerDock') and self.viewerDock: 
            self.viewerDock.toggleViewAction().setText("Toggle System Viewer") # Ensure text
            viewMenu.addAction(self.viewerDock.toggleViewAction())
        if hasattr(self, 'lensEditorDock') and self.lensEditorDock: 
            self.lensEditorDock.toggleViewAction().setText("Toggle Lens Data Editor")
            viewMenu.addAction(self.lensEditorDock.toggleViewAction())
        if hasattr(self, 'systemPropertiesDock') and self.systemPropertiesDock: 
            self.systemPropertiesDock.toggleViewAction().setText("Toggle System Properties")
            viewMenu.addAction(self.systemPropertiesDock.toggleViewAction())
        if hasattr(self, 'analysisPanelDock') and self.analysisPanelDock: 
            self.analysisPanelDock.toggleViewAction().setText("Toggle Analysis Panel")
            viewMenu.addAction(self.analysisPanelDock.toggleViewAction())
        if hasattr(self, 'optimizationPanelDock') and self.optimizationPanelDock: 
            self.optimizationPanelDock.toggleViewAction().setText("Toggle Optimization Panel")
            viewMenu.addAction(self.optimizationPanelDock.toggleViewAction())

        runMenu = menu_bar.addMenu("&Run")
        helpMenu = menu_bar.addMenu("&Help")
        helpMenu.addAction(self.aboutAction)

    def _populate_quick_actions_toolbar(self, toolbar: QToolBar):
        """Populates the 'Quick Actions' toolbar."""
        toolbar.addAction(self.newAction)
        toolbar.addAction(self.openAction)
        toolbar.addAction(self.saveAction)
        toolbar.addSeparator()
        toolbar.addAction(self.loadLayout1Action)
        toolbar.addAction(self.loadLayout2Action)
        toolbar.addAction(self.saveLayoutAction)
        toolbar.addSeparator()
        toolbar.addAction(self.dockAllAction)
        toolbar.addAction(self.resetLayoutAction)

    def _create_actions(self):
        self.newAction = QAction("&New System",self,shortcut=QKeySequence.New,triggered=self.new_system_action)
        self.openAction = QAction("&Open System...",self,shortcut=QKeySequence.Open,triggered=self.open_system_action)
        self.saveAction = QAction("&Save System",self,shortcut=QKeySequence.Save,triggered=self.save_system_action)
        self.saveAsAction = QAction("Save System &As...",self,shortcut=QKeySequence.SaveAs,triggered=self.save_system_as_action)
        self.exitAction = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)

        self.loadLayout1Action = QAction("1", self, triggered=self.load_layout_1_slot)
        self.loadLayout1Action.setToolTip("Load Layout from Slot 1")
        self.loadLayout2Action = QAction("2", self, triggered=self.load_layout_2_slot)
        self.loadLayout2Action.setToolTip("Load Layout from Slot 2")
        self.saveLayoutAction = QAction("Save Current Layout", self, triggered=self.save_layout_slot)
        self.saveLayoutAction.setToolTip("Save current window layout to next available slot (1 or 2)")
        self.loadLayout1Action.setEnabled(self.settings.contains("Layouts/Config1Geometry"))
        self.loadLayout2Action.setEnabled(self.settings.contains("Layouts/Config2Geometry"))

        self.dockAllAction = QAction("Dock All Windows",self,triggered=self.dock_all_windows_action)
        self.resetLayoutAction = QAction("Reset Window Layout",self,triggered=self.reset_windows_action)
        
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

    def _create_menu_bar(self): # Obsolete
        pass

    def _create_tool_bars(self): # Obsolete
        pass

    def _handle_maximize_restore(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def changeEvent(self, event: QEvent):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            # Update the custom title bar's maximize button state
            if hasattr(self, 'custom_title_bar_widget') and self.custom_title_bar_widget:
                self.custom_title_bar_widget.update_maximize_button_state(self.isMaximized())
    
    def load_stylesheets(self):
        style_str = ""
        try:
            with open(self.current_theme_path, 'r') as f_theme:
                style_str += f_theme.read()
            print(f"Main theme loaded: {self.current_theme_path}")
        except Exception as e:
            print(f"Error loading main theme {self.current_theme_path}: {e}")

        if os.path.exists(SIDEBAR_QSS_PATH):
            try:
                with open(SIDEBAR_QSS_PATH, 'r') as f_sidebar:
                    style_str += "\n" + f_sidebar.read()
                print(f"Sidebar stylesheet loaded: {SIDEBAR_QSS_PATH}")
            except Exception as e:
                print(f"Error loading sidebar stylesheet {SIDEBAR_QSS_PATH}: {e}")
        
        if os.path.exists(CUSTOM_TITLE_BAR_QSS_PATH):
            try:
                with open(CUSTOM_TITLE_BAR_QSS_PATH, 'r') as f_titlebar:
                    style_str += "\n" + f_titlebar.read()
                print(f"Custom title bar stylesheet loaded: {CUSTOM_TITLE_BAR_QSS_PATH}")
            except Exception as e:
                print(f"Error loading custom title bar stylesheet {CUSTOM_TITLE_BAR_QSS_PATH}: {e}")

        self.setStyleSheet(style_str) 
        
        # Also apply to custom_title_bar_widget if it doesn't inherit styles properly
        # or if it needs specific top-level styling not covered by its children.
        if hasattr(self, 'custom_title_bar_widget'):
             self.custom_title_bar_widget.setStyleSheet(style_str)


        if hasattr(self, "darkThemeAction") and hasattr(self, "lightThemeAction"):
            is_dark = (self.current_theme_path == THEME_DARK_PATH)
            self.darkThemeAction.setChecked(is_dark)
            self.lightThemeAction.setChecked(not is_dark)

    def _update_project_name_in_title_bar(self):
        # Update the project name in the custom_title_bar_widget
        if hasattr(self, 'custom_title_bar_widget') and self.custom_title_bar_widget:
            project_name = "UnnamedProject.opds" 
            optic = self.connector.get_optic()
            if optic and optic.name: 
                current_file = self.connector.get_current_filepath()
                if current_file:
                    project_name = os.path.basename(current_file)
                elif optic.name != "Default System" and optic.name != "New Untitled System":
                     project_name = optic.name + ".opds (unsaved)" 
                else:
                    project_name = "UnnamedProject.opds"
            self.custom_title_bar_widget.set_project_name(project_name)

    def animate_dock_toggle(self, dock_widget, show_state_after_toggle):
        animation_duration = 300 
        easing_curve = QEasingCurve.InOutQuad
        is_left_or_right_dock = self.dockWidgetArea(dock_widget) in [Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.RightDockWidgetArea,]
        default_w = 300 
        default_h = 200 
        
        original_dimension = 0
        if dock_widget == self.sidebarDock:
            if hasattr(self, 'sidebar_content_widget') and self.sidebar_content_widget: # Check existence
                 original_dimension = self.sidebar_content_widget.maximumWidth() if not self.sidebar_content_widget._is_collapsed else self.sidebar_content_widget.minimumWidth()
            else: # Fallback if sidebar_content_widget is not yet available
                original_dimension = SIDEBAR_MAX_WIDTH 
        elif is_left_or_right_dock:
            original_dimension = self.dock_original_sizes.get(dock_widget, dock_widget.width() if dock_widget.width() > 0 else default_w) 
        else: 
            original_dimension = self.dock_original_sizes.get(dock_widget, dock_widget.height() if dock_widget.height() > 0 else default_h) 

        if (dock_widget in self.dock_animations and self.dock_animations[dock_widget].state() == QPropertyAnimation.Running):
            self.dock_animations[dock_widget].stop()
        current_visibility = not dock_widget.isHidden()
        if show_state_after_toggle:
            if not current_visibility:
                dock_widget.show()
                dock_widget.raise_()
                target_prop = b"maximumWidth" if is_left_or_right_dock else b"maximumHeight"
                animation = QPropertyAnimation(dock_widget, target_prop)
                animation.setStartValue(0)
                animation.setEndValue(original_dimension) 
                if is_left_or_right_dock: dock_widget.setMaximumWidth(original_dimension if original_dimension > 0 else 5000) 
                else: dock_widget.setMaximumHeight(original_dimension if original_dimension > 0 else 5000)
                animation.setDuration(animation_duration)
                animation.setEasingCurve(easing_curve)
                animation.start(QPropertyAnimation.DeleteWhenStopped)
                self.dock_animations[dock_widget] = animation
            else: 
                dock_widget.raise_()
                if is_left_or_right_dock: dock_widget.setMaximumWidth(original_dimension if original_dimension > 0 else 5000)
                else: dock_widget.setMaximumHeight(original_dimension if original_dimension > 0 else 5000)
        else: 
            if current_visibility:
                current_size = dock_widget.width() if is_left_or_right_dock else dock_widget.height()
                target_prop = b"maximumWidth" if is_left_or_right_dock else b"maximumHeight"
                animation = QPropertyAnimation(dock_widget, target_prop)
                animation.setStartValue(current_size)
                animation.setEndValue(0)
                animation.setDuration(animation_duration)
                animation.setEasingCurve(easing_curve)
                animation.finished.connect(dock_widget.hide)
                animation.finished.connect(
                    lambda: dock_widget.setMaximumWidth(original_dimension if original_dimension > 0 else 5000) 
                    if is_left_or_right_dock else 
                    dock_widget.setMaximumHeight(original_dimension if original_dimension > 0 else 5000)
                )
                animation.start(QPropertyAnimation.DeleteWhenStopped)
                self.dock_animations[dock_widget] = animation
    
    @Slot(str)
    def switch_theme(self, theme_path):
        if theme_path != self.current_theme_path:
            self.current_theme_path = theme_path
            self.load_stylesheets()

    @Slot()
    def new_system_action(self):
        self.connector.new_system()
        self._update_project_name_in_title_bar()
        print("Main Window: New System action triggered")

    @Slot()
    def open_system_action(self):
        filepath, _ = QFileDialog.getOpenFileName(self,"Open Optiland System","", "Optiland JSON Files (*.json);;All Files (*)")
        if filepath:
            self.connector.load_optic_from_file(filepath)
            self._update_project_name_in_title_bar()
            print(f"Main Window: Open System action triggered - {filepath}")

    @Slot()
    def save_system_action(self):
        current_path = self.connector.get_current_filepath()
        if current_path:
            self.connector.save_optic_to_file(current_path)
            self._update_project_name_in_title_bar() 
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
            self._update_project_name_in_title_bar()
            print(f"Main Window: Save System As action triggered - {filepath}")

    @Slot()
    def about_action(self):
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("About Optiland GUI")
        layout = QVBoxLayout(about_dialog)
        about_text = QLabel(
            "<p><b>Optiland GUI</b></p>"
            "<p>A modern interface for the Optiland optical simulation package.</p>"
            "<p>Version: 0.2.1 (Frameless Layout Refined)</p>" 
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
        print("Main Window: Dock All Windows action triggered - resetting to revised default layout.")
        self._apply_revised_default_dock_layout()
        for dock in self.all_managed_docks:
            if dock and dock.toggleViewAction():
                dock.toggleViewAction().setChecked(True) 

    @Slot()
    def reset_windows_action(self):
        print("Main Window: Reset Windows action triggered - resetting to revised default layout.")
        self._apply_revised_default_dock_layout()
        for dock in self.all_managed_docks: 
            if dock and dock.toggleViewAction(): 
                dock.toggleViewAction().setChecked(True)

    @Slot()
    def save_layout_slot(self):
        target_slot = self.next_save_slot_index
        window_geometry = self.saveGeometry()
        dock_toolbar_state = self.saveState() 
        self.settings.setValue(f"Layouts/Config{target_slot}Geometry", window_geometry)
        self.settings.setValue(f"Layouts/Config{target_slot}State", dock_toolbar_state)
        QMessageBox.information(self, "Layout Saved", f"The current window layout was saved to configuration - {target_slot}")
        self.next_save_slot_index = 2 if target_slot == 1 else 1
        self.settings.setValue("Layouts/NextSaveSlot", self.next_save_slot_index)
        self.loadLayout1Action.setEnabled(self.settings.contains("Layouts/Config1Geometry"))
        self.loadLayout2Action.setEnabled(self.settings.contains("Layouts/Config2Geometry"))
        print(f"Layout saved to slot {target_slot}. Next save will be to slot {self.next_save_slot_index}.")

    def _load_layout_from_slot(self, slot_number):
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
                QMessageBox.information(self, "Layout Loaded", f"Layout from configuration - {slot_number} has been loaded.")
            else:
                QMessageBox.warning(self, "Load Error", f"Invalid layout data found in configuration - {slot_number}.")
        else:
            QMessageBox.information(self, "Load Layout", f"No layout saved in configuration - {slot_number}.")

    @Slot()
    def load_layout_1_slot(self):
        print("Loading layout from slot 1...")
        self._load_layout_from_slot(1)

    @Slot()
    def load_layout_2_slot(self):
        print("Loading layout from slot 2...")
        self._load_layout_from_slot(2)

    def closeEvent(self, event: QEvent):
        print("Main Window: Closing application.")
        event.accept()