"""Defines the main window of the Optiland GUI application.

This module contains the `MainWindow` class, which serves as the main entry point
and container for all GUI elements, including the lens editor, analysis panels,
viewers, and toolbars. It manages window layout, themes, actions, and the
connection to the backend via the `OptilandConnector`.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

import contextlib
import inspect
import os
from collections import defaultdict

from PySide6.QtCore import (
    QByteArray,
    QEasingCurve,
    QEvent,
    QPoint,
    QPropertyAnimation,
    QRect,
    QSettings,
    Qt,
    Slot,
)
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QPainter, QResizeEvent
from PySide6.QtWidgets import (
    QDialog,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QStyle,
    QStyleOption,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import optiland.samples
from optiland.optic import Optic

from . import gui_plot_utils
from .analysis_panel import AnalysisPanel
from .lens_editor import LensEditor
from .optiland_connector import OptilandConnector

# from .optimization_panel import OptimizationPanel # we will support this later on
from .system_properties_panel import SystemPropertiesPanel
from .viewer_panel import ViewerPanel
from .widgets.custom_title_bar import CustomTitleBar
from .widgets.python_terminal import PythonTerminalWidget
from .widgets.sidebar import (
    SIDEBAR_MAX_WIDTH,
    SIDEBAR_MIN_WIDTH,
    SidebarWidget,
)

try:
    from .resources import resources_rc  # noqa: F401
except ImportError as e:
    print(f"Warning (main_window.py): Could not import resources_rc.py: {e}")

THEME_DARK_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "styles", "dark_theme.qss"
)
THEME_LIGHT_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "styles", "light_theme.qss"
)
SIDEBAR_QSS_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "styles", "sidebar.qss"
)

ORGANIZATION_NAME = "OptilandProject"
APPLICATION_NAME = "OptilandGUI"


class CustomDockTitleBar(QWidget):
    """A custom title bar for QDockWidgets with macOS-style buttons."""

    def __init__(self, parent_dock: QDockWidget, title: str = ""):
        super().__init__(parent_dock)
        self.setObjectName("CustomDockTitleBar")
        self.dock_widget = parent_dock  # Store reference to the actual dock widget

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 3, 8, 3)  # Left, Top, Right, Bottom
        layout.setSpacing(2)

        self.title_label = QLabel(title)
        layout.addWidget(self.title_label)
        layout.addStretch()

        # Create buttons
        self.minimize_btn = QPushButton(self)
        self.minimize_btn.setObjectName("DockMinimizeButton")
        self.minimize_btn.setFixedSize(10, 10)
        self.minimize_btn.setToolTip("Hide")
        self.minimize_btn.clicked.connect(parent_dock.toggleViewAction().trigger)

        self.undock_btn = QPushButton(self)
        self.undock_btn.setObjectName("DockUndockButton")
        self.undock_btn.setFixedSize(10, 10)
        self.undock_btn.setToolTip("Float/Dock")
        self.undock_btn.clicked.connect(
            lambda: parent_dock.setFloating(not parent_dock.isFloating())
        )

        self.close_btn = QPushButton(self)
        self.close_btn.setObjectName("DockCloseButton")
        self.close_btn.setFixedSize(10, 10)
        self.close_btn.setToolTip("Close")
        self.close_btn.clicked.connect(parent_dock.close)

        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.undock_btn)
        layout.addWidget(self.close_btn)

    def paintEvent(self, event):
        """Ensures the background is drawn correctly according to the stylesheet."""
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(
            QStyle.PrimitiveElement.PE_Widget, opt, painter, self
        )
        super().paintEvent(event)

    def mousePressEvent(self, event):
        """Handles mouse press for dragging the floating dock."""
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.dock_widget.isFloating()
        ):
            self._mouse_press_pos = event.globalPosition().toPoint()
            self._window_pos_before_move = self.dock_widget.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handles mouse move for dragging."""
        if (
            self.dock_widget.isFloating()
            and event.buttons() == Qt.MouseButton.LeftButton
        ) and (hasattr(self, "_mouse_press_pos") and self._mouse_press_pos is not None):
            delta = event.globalPosition().toPoint() - self._mouse_press_pos
            self.dock_widget.move(self._window_pos_before_move + delta)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handles mouse release to stop dragging."""
        self._mouse_press_pos = None
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    """The main application window for the Optiland GUI.

    This class orchestrates the entire graphical user interface. It initializes
    and manages all dockable widgets (like the Lens Editor and Analysis Panel),
    handles user actions through menus and toolbars, manages window layout and
    theming, and provides a scripting interface to control the application.

    Attributes:
        connector (OptilandConnector): The central connector for backend communication.
        iface (OptilandInterface): The scripting interface exposed to the Python
                                    console.
        all_managed_docks (list): A list of all dock widgets managed by the main
                                    window.
    """

    class OptilandInterface:
        """A high-level interface for controlling the Optiland GUI via scripting.

        This object is made available in the integrated Python console as 'iface',
        allowing users to programmatically interact with the main application
        components, such as opening panels, refreshing views, and accessing data.

        Args:
            main_window (MainWindow): A reference to the main application window.
        """

        def __init__(self, main_window):
            self._win = main_window

        def get_main_window(self):
            """Returns the main application window instance.

            Returns:
                MainWindow: The main QMainWindow instance.
            """
            return self._win

        def get_analysis_panel(self):
            """Returns the primary AnalysisPanel widget instance.

            Returns:
                AnalysisPanel: The main analysis panel widget.
            """
            return self._win.analysisPanel

        def get_lens_editor(self):
            """Returns the LensEditor widget instance.

            Returns:
                LensEditor: The lens data editor widget.
            """
            return self._win.lensEditor

        def get_viewer_panel(self):
            """Returns the ViewerPanel widget instance.

            Returns:
                ViewerPanel: The 2D/3D viewer panel widget.
            """
            return self._win.viewerPanel

        def show_lens_editor(self):
            """Brings the Lens Data Editor dock widget to the front."""
            self._win.lensEditorDock.show()
            self._win.lensEditorDock.raise_()

        def show_analysis_panel(self):
            """Brings the Analysis Panel dock widget to the front."""
            self._win.analysisPanelDock.show()
            self._win.analysisPanelDock.raise_()
            # Also ensure its tab is selected
            parent_tab_widget = self._win.analysisPanelDock.parentWidget()
            if isinstance(parent_tab_widget, QTabWidget):
                parent_tab_widget.setCurrentWidget(self._win.analysisPanelDock)

        def refresh_all(self):
            """Triggers a full refresh of all GUI panels.

            This is a convenience method that emits the `opticChanged` signal from
            the connector, prompting all connected widgets to reload their data.
            """
            print("GUI refresh requested via iface.refresh_all()")
            self._win.connector.opticChanged.emit()

    def __init__(self):
        """Initializes the MainWindow by orchestrating the setup of all
        UI components."""
        super().__init__()
        self._configure_window()
        self._init_core_components()
        self._init_ui()
        self._finalize_setup()

    def _init_ui(self):
        """Initializes the main UI components, panels, docks, and toolbars."""
        self._init_ui_panels()
        self._setup_all_dock_widgets()
        self._create_actions()
        self._setup_menus_and_toolbars()
        self._setup_layout()

    def _configure_window(self):
        """Sets up the main window's flags, title, geometry, and resizing logic."""
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setWindowTitle("Optiland GUI")
        self.setGeometry(100, 100, 1600, 900)
        self.setMouseTracking(True)
        self.grip_size = 8
        self.is_resizing = False
        self.resize_area = None
        self.is_moving = False
        self.drag_position = QPoint()
        self.start_geometry = None
        self.start_pos = None

    def _init_core_components(self):
        """Initializes non-UI core components like settings, the connector,
        and the scripting interface."""
        self.current_theme_path = THEME_DARK_PATH
        self.analysis_panels = []
        self.settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)
        self.next_save_slot_index = self.settings.value(
            "Layouts/NextSaveSlot", 1, type=int
        )
        self.connector = OptilandConnector()
        self.iface = self.OptilandInterface(self)
        self.dock_animations = {}
        self.dock_original_sizes = {}
        self.about_dialog = None

    def _init_ui_panels(self):
        """Creates instances of all the main UI panels and widgets."""
        self.lensEditor = LensEditor(self.connector)
        self.viewerPanel = ViewerPanel(self.connector)
        self.analysisPanel = AnalysisPanel(self.connector)
        self.systemPropertiesPanel = SystemPropertiesPanel(self.connector)
        initial_theme_name = (
            "dark" if self.current_theme_path == THEME_DARK_PATH else "light"
        )
        self.pythonTerminal = PythonTerminalWidget(
            self,
            custom_variables={"connector": self.connector, "iface": self.iface},
            theme=initial_theme_name,
        )
        self.pythonTerminal.commandExecuted.connect(self.refresh_all_gui_panels)

    def _setup_menus_and_toolbars(self):
        """Creates and populates the main menu bar, custom title bar, and toolbars."""
        # Main Menu Bar
        self._actual_menu_bar_instance = QMenuBar(self)
        self._populate_main_menu_bar(self._actual_menu_bar_instance)

        # Custom Title Bar
        self.custom_title_bar_widget = CustomTitleBar(
            self._actual_menu_bar_instance, self
        )
        self.custom_title_bar_widget.minimize_requested.connect(self.showMinimized)
        self.custom_title_bar_widget.maximize_restore_requested.connect(
            self._handle_maximize_restore
        )
        self.custom_title_bar_widget.close_requested.connect(self.close)
        self.custom_title_bar_widget.settings_requested.connect(self.show_settings_wip)

        self.title_bar_as_toolbar = QToolBar("CustomTitleBarToolbar")
        self.title_bar_as_toolbar.setObjectName("CustomTitleBarToolbar")
        self.title_bar_as_toolbar.setMovable(False)
        self.title_bar_as_toolbar.setFloatable(False)
        self.title_bar_as_toolbar.addWidget(self.custom_title_bar_widget)
        self.addToolBar(Qt.TopToolBarArea, self.title_bar_as_toolbar)

        # Quick Actions Toolbar
        self.quick_actions_toolbar = QToolBar("QuickActionsToolbar")
        self.quick_actions_toolbar.setObjectName("QuickActionsToolbar")
        self.quick_actions_toolbar.setMovable(True)
        self._populate_quick_actions_toolbar(self.quick_actions_toolbar)
        self.addToolBarBreak(Qt.TopToolBarArea)
        self.addToolBar(Qt.TopToolBarArea, self.quick_actions_toolbar)

    def _setup_layout(self):
        """Sets up the central widget and the default dock layout."""
        self.main_docking_area_placeholder = QWidget()
        self.main_docking_area_placeholder.setObjectName("MainDockingAreaPlaceholder")
        self.setCentralWidget(self.main_docking_area_placeholder)
        self.setDockNestingEnabled(True)
        self._apply_revised_default_dock_layout()

    def _finalize_setup(self):
        """Applies stylesheets, connects signals, and sets the initial UI state."""
        self.load_stylesheets()
        if hasattr(self, "darkThemeAction"):
            self.darkThemeAction.setChecked(self.current_theme_path == THEME_DARK_PATH)
            self.lightThemeAction.setChecked(self.current_theme_path != THEME_DARK_PATH)

        self._connect_dock_animations()

        self.connector.opticLoaded.connect(self._update_project_name_in_title_bar)
        self.connector.opticChanged.connect(self._update_project_name_in_title_bar)
        self.connector.modifiedStateChanged.connect(
            self._update_project_name_in_title_bar
        )

        self._initial_narrow_check_done = False
        self._update_project_name_in_title_bar()

    def _setup_all_dock_widgets(self):
        """Initializes and configures all dockable widgets for the application.

        This method creates instances of all the main panels (Sidebar, Viewer,
        Lens Editor, etc.) and wraps them in `QDockWidget` containers. It sets
        their properties, such as features and object names.
        """
        self.sidebar_content_widget = SidebarWidget(self)
        self.sidebar_content_widget.menuSelected.connect(self.on_sidebar_menu_selected)
        self.sidebarDock = QDockWidget("NavigationSidebar", self)
        self.sidebarDock.setWidget(self.sidebar_content_widget)
        self.sidebarDock.setObjectName("SidebarDockWidget")
        self.sidebarDock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.sidebarDock.setTitleBarWidget(QWidget())
        self.sidebarDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.viewerDock = QDockWidget("System Viewer", self)
        self.viewerDock.setWidget(self.viewerPanel)
        self.viewerDock.setObjectName("ViewerDock")
        self.viewerDock.setTitleBarWidget(
            CustomDockTitleBar(self.viewerDock, "System Viewer")
        )
        self.viewerDock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

        self.lensEditorDock = QDockWidget("Lens Data Editor", self)
        self.lensEditorDock.setWidget(self.lensEditor)
        self.lensEditorDock.setObjectName("LensEditorDock")
        self.lensEditorDock.setTitleBarWidget(
            CustomDockTitleBar(self.lensEditorDock, "Lens Data Editor")
        )
        self.lensEditorDock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

        self.systemPropertiesDock = QDockWidget("System Properties", self)
        self.systemPropertiesDock.setWidget(self.systemPropertiesPanel)
        self.systemPropertiesDock.setObjectName("SystemPropertiesDock")
        self.systemPropertiesDock.setTitleBarWidget(
            CustomDockTitleBar(self.systemPropertiesDock, "System Properties")
        )
        self.systemPropertiesDock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

        self.analysisPanelDock = QDockWidget("Analysis", self)
        self.analysisPanelDock.setWidget(self.analysisPanel)
        self.analysisPanelDock.setObjectName("AnalysisPanelDock")
        self.analysisPanelDock.setTitleBarWidget(
            CustomDockTitleBar(self.analysisPanelDock, "Analysis")
        )
        self.analysisPanelDock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        self.analysisPanelDock.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.analysisPanelDock.customContextMenuRequested.connect(
            lambda pos: self._show_dock_context_menu(pos, self.analysisPanelDock)
        )

        self.terminalDock = QDockWidget("Scripts Terminal", self)
        self.terminalDock.setWidget(self.pythonTerminal)
        self.terminalDock.setObjectName("TerminalDock")
        self.terminalDock.setTitleBarWidget(
            CustomDockTitleBar(self.terminalDock, "Scripts Terminal")
        )
        self.terminalDock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

        self.all_managed_docks = [
            self.sidebarDock,
            self.viewerDock,
            self.lensEditorDock,
            self.systemPropertiesDock,
            self.analysisPanelDock,
            # self.optimizationPanelDock,
            # we will support the optimiziation feature later on
            self.terminalDock,
        ]

    def _apply_revised_default_dock_layout(self):
        """Applies the default docking layout to the main window.

        This function arranges the dock widgets in a predefined layout, splitting
        and tabbing them to create a functional and organized user interface.
        This is called on first launch and when resetting the layout.
        """
        for dock in self.all_managed_docks:
            if dock:
                dock.setFloating(False)
                dock.show()

        # 1. Place the Sidebar on the far left.
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.sidebarDock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.lensEditorDock)
        self.splitDockWidget(
            self.lensEditorDock, self.analysisPanelDock, Qt.Orientation.Horizontal
        )
        self.splitDockWidget(
            self.lensEditorDock, self.viewerDock, Qt.Orientation.Vertical
        )
        self.splitDockWidget(
            self.analysisPanelDock, self.terminalDock, Qt.Orientation.Vertical
        )
        # self.tabifyDockWidget(self.analysisPanelDock, self.optimizationPanelDock)
        # we will support this later on
        self.tabifyDockWidget(self.analysisPanelDock, self.systemPropertiesDock)

        self.sidebarDock.raise_()
        self.lensEditorDock.raise_()
        self.systemPropertiesDock.raise_()
        self.viewerDock.raise_()
        self.analysisPanelDock.raise_()

        if self.viewerPanel.viewer2D and hasattr(
            self.viewerPanel.viewer2D, "plot_optic"
        ):
            self.viewerPanel.viewer2D.plot_optic()
        if self.viewerPanel.viewer3D and hasattr(
            self.viewerPanel.viewer3D, "render_optic"
        ):
            self.viewerPanel.viewer3D.render_optic()

    def showEvent(self, event: QResizeEvent):
        """Handles the window show event.

        This overridden method performs initial setup tasks the first time the
        window is shown, such as adjusting the sidebar's collapsed state based
        on the initial window width.

        Args:
            event: The QShowEvent.
        """
        super().showEvent(event)
        if not self._initial_narrow_check_done:
            if hasattr(self, "sidebar_content_widget") and self.sidebar_content_widget:
                if self.width() < (SIDEBAR_MAX_WIDTH + 300):
                    self.sidebar_content_widget.force_set_collapse_state(True)
                    if self.sidebarDock.width() > SIDEBAR_MIN_WIDTH:
                        self.resizeDocks(
                            [self.sidebarDock], [SIDEBAR_MIN_WIDTH], Qt.Horizontal
                        )
                else:
                    self.sidebar_content_widget.force_set_collapse_state(False)
                    if self.sidebarDock.width() < 150:
                        self.resizeDocks([self.sidebarDock], [150], Qt.Horizontal)
            self._initial_narrow_check_done = True

        if hasattr(self, "darkThemeAction") and hasattr(self, "lightThemeAction"):
            is_dark = self.current_theme_path == THEME_DARK_PATH
            self.darkThemeAction.setChecked(is_dark)
            self.lightThemeAction.setChecked(not is_dark)

    def _connect_dock_animations(self):
        """Connects dock widget view actions to an animation handler.

        This method disconnects the default `triggered` signal from each dock's
        toggle view action and reconnects it to a custom slot that provides a
        fade-in/out animation for a smoother user experience.
        """
        if not hasattr(self, "all_managed_docks"):
            return
        for dock_widget_ref in self.all_managed_docks:
            if dock_widget_ref:
                action = dock_widget_ref.toggleViewAction()
                if action:
                    with contextlib.suppress(TypeError, RuntimeError):
                        action.triggered.disconnect()
                    action.triggered.connect(
                        lambda checked, dock=dock_widget_ref: self.animate_dock_toggle(
                            dock, checked
                        )
                    )

    def _clone_analysis_window(self):
        """Creates a new, independent AnalysisPanel in a floating dock widget.

        This allows the user to have multiple analysis windows open simultaneously,
        for example, to compare results with different settings.
        """
        cloned_panel = AnalysisPanel(self.connector)
        cloned_panel.update_theme_icons(self.current_theme_path)

        new_dock = QDockWidget("Analysis-Cloned", self)
        new_dock.setObjectName("ClonedAnalysisDock")
        new_dock.setWidget(cloned_panel)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, new_dock)
        new_dock.setFloating(True)
        new_dock.resize(self.analysisPanelDock.size())

        new_dock.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        new_dock.customContextMenuRequested.connect(
            lambda pos, dock=new_dock: self._show_dock_context_menu(pos, dock)
        )

    def _show_dock_context_menu(self, position, dock_widget):
        """Creates and shows a right-click context menu for a dock widget.

        This menu provides standard actions like docking/undocking and custom
        actions like cloning the window.

        Args:
            position: The position where the right-click occurred.
            dock_widget: The QDockWidget that was right-clicked.
        """
        menu = QMenu()

        menu.addAction(dock_widget.toggleViewAction())
        menu.addSeparator()
        clone_action = menu.addAction("Clone Window")

        action = menu.exec(dock_widget.mapToGlobal(position))

        if action == clone_action and isinstance(dock_widget.widget(), AnalysisPanel):
            self._clone_analysis_window()

    def _populate_main_menu_bar(self, menu_bar: QMenuBar):
        """Populates the main menu bar with actions and sub-menus.

        Args:
            menu_bar: The QMenuBar instance to populate.
        """
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

        if hasattr(self, "terminalDock") and self.terminalDock:
            self.terminalDock.toggleViewAction().setText("Toggle Scripts Terminal")
            viewMenu.addAction(self.terminalDock.toggleViewAction())

        for dock in self.all_managed_docks:
            if dock and dock.toggleViewAction():
                dock.toggleViewAction().setText(f"Toggle {dock.windowTitle()}")
                viewMenu.addAction(dock.toggleViewAction())

        self._populate_gallery_menu(menu_bar)

        menu_bar.addMenu("&Run")
        helpMenu = menu_bar.addMenu("&Help")
        helpMenu.addAction(self.aboutAction)

    def _populate_quick_actions_toolbar(self, toolbar: QToolBar):
        """Populates the quick actions toolbar with common actions.

        Args:
            toolbar: The QToolBar instance to populate.
        """
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
        """Creates all QAction objects used in menus and toolbars."""
        self.newAction = QAction(
            "&New System",
            self,
            shortcut=QKeySequence.New,
            triggered=self.new_system_action,
        )
        self.openAction = QAction(
            "&Open System...",
            self,
            shortcut=QKeySequence.Open,
            triggered=self.open_system_action,
        )
        self.saveAction = QAction(
            "&Save System",
            self,
            shortcut=QKeySequence.Save,
            triggered=self.save_system_action,
        )
        self.saveAsAction = QAction(
            "Save System &As...",
            self,
            shortcut=QKeySequence.SaveAs,
            triggered=self.save_system_as_action,
        )
        self.exitAction = QAction(
            "E&xit", self, shortcut="Ctrl+Q", triggered=self.close
        )

        self.loadLayout1Action = QAction("1", self, triggered=self.load_layout_1_slot)
        self.loadLayout1Action.setToolTip("Load Layout from Slot 1")
        self.loadLayout2Action = QAction("2", self, triggered=self.load_layout_2_slot)
        self.loadLayout2Action.setToolTip("Load Layout from Slot 2")
        self.saveLayoutAction = QAction(
            "Save Current Layout", self, triggered=self.save_layout_slot
        )
        self.saveLayoutAction.setToolTip(
            "Save current window layout to next available slot (1 or 2)"
        )
        self.loadLayout1Action.setEnabled(
            self.settings.contains("Layouts/Config1Geometry")
        )
        self.loadLayout2Action.setEnabled(
            self.settings.contains("Layouts/Config2Geometry")
        )

        self.dockAllAction = QAction(
            "Dock All Windows", self, triggered=self.reset_windows_action
        )
        self.resetLayoutAction = QAction(
            "Reset Window Layout", self, triggered=self.reset_windows_action
        )

        self.themeActionGroup = QActionGroup(self)
        self.themeActionGroup.setExclusive(True)
        self.darkThemeAction = QAction("Dark Theme", self, checkable=True)
        self.darkThemeAction.triggered.connect(
            lambda: self.switch_theme(THEME_DARK_PATH)
        )
        self.themeActionGroup.addAction(self.darkThemeAction)
        self.lightThemeAction = QAction("Light Theme", self, checkable=True)
        self.lightThemeAction.triggered.connect(
            lambda: self.switch_theme(THEME_LIGHT_PATH)
        )
        self.themeActionGroup.addAction(self.lightThemeAction)

        self.aboutAction = QAction(
            "&About Optiland GUI", self, triggered=self.about_action
        )
        self.undoAction = QAction(
            "&Undo", self, shortcut=QKeySequence.Undo, triggered=self.connector.undo
        )
        self.undoAction.setEnabled(False)
        self.redoAction = QAction(
            "&Redo", self, shortcut=QKeySequence.Redo, triggered=self.connector.redo
        )
        self.redoAction.setEnabled(False)
        self.connector.undoStackAvailabilityChanged.connect(self.undoAction.setEnabled)
        self.connector.redoStackAvailabilityChanged.connect(self.redoAction.setEnabled)

    def _handle_maximize_restore(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def changeEvent(self, event: QEvent):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and (
            hasattr(self, "custom_title_bar_widget") and self.custom_title_bar_widget
        ):
            self.custom_title_bar_widget.update_maximize_button_state(
                self.isMaximized()
            )

    def load_stylesheets(self):
        style_str = ""
        try:
            with open(self.current_theme_path) as f_theme:
                style_str += f_theme.read()
        except Exception as e:
            print(f"Error loading main theme {self.current_theme_path}: {e}")

        if os.path.exists(SIDEBAR_QSS_PATH):
            try:
                with open(SIDEBAR_QSS_PATH) as f_sidebar:
                    style_str += "\n" + f_sidebar.read()

            except Exception as e:
                print(f"Error loading sidebar stylesheet {SIDEBAR_QSS_PATH}: {e}")

        self.setStyleSheet(style_str)

        theme_name = "dark" if self.current_theme_path == THEME_DARK_PATH else "light"
        gui_plot_utils.apply_gui_matplotlib_styles(theme=theme_name)

        if hasattr(self, "custom_title_bar_widget"):
            self.custom_title_bar_widget.setStyleSheet(style_str)

        if hasattr(self, "darkThemeAction") and hasattr(self, "lightThemeAction"):
            is_dark = self.current_theme_path == THEME_DARK_PATH
            self.darkThemeAction.setChecked(is_dark)
            self.lightThemeAction.setChecked(not is_dark)

    def _update_project_name_in_title_bar(self):
        if hasattr(self, "custom_title_bar_widget") and self.custom_title_bar_widget:
            # New logic starts here
            display_name = "UnnamedProject.json"
            current_file = self.connector.get_current_filepath()
            is_modified = self.connector.is_modified()

            if current_file:
                display_name = os.path.basename(current_file)

            if not current_file:
                is_modified = True

            if is_modified:
                display_name += "*"

            self.custom_title_bar_widget.set_project_name(display_name)

    def _animate_dock_show(
        self, dock_widget, is_left_or_right, original_dimension, duration, curve
    ):
        """Handles the animation for showing a dock widget."""
        if dock_widget.isHidden():
            dock_widget.show()
            dock_widget.raise_()
            target_prop = b"maximumWidth" if is_left_or_right else b"maximumHeight"
            animation = QPropertyAnimation(dock_widget, target_prop)
            animation.setStartValue(0)
            animation.setEndValue(original_dimension)
            if is_left_or_right:
                dock_widget.setMaximumWidth(
                    original_dimension if original_dimension > 0 else 5000
                )
            else:
                dock_widget.setMaximumHeight(
                    original_dimension if original_dimension > 0 else 5000
                )
            animation.setDuration(duration)
            animation.setEasingCurve(curve)
            animation.start(QPropertyAnimation.DeleteWhenStopped)
            self.dock_animations[dock_widget] = animation
        else:
            dock_widget.raise_()

    def _animate_dock_hide(
        self, dock_widget, is_left_or_right, original_dimension, duration, curve
    ):
        """Handles the animation for hiding a dock widget."""
        if not dock_widget.isHidden():
            current_size = (
                dock_widget.width() if is_left_or_right else dock_widget.height()
            )
            target_prop = b"maximumWidth" if is_left_or_right else b"maximumHeight"
            animation = QPropertyAnimation(dock_widget, target_prop)
            animation.setStartValue(current_size)
            animation.setEndValue(0)
            animation.setDuration(duration)
            animation.setEasingCurve(curve)
            animation.finished.connect(dock_widget.hide)
            animation.finished.connect(
                lambda: dock_widget.setMaximumSize(5000, 5000)
            )  # Restore max size
            animation.start(QPropertyAnimation.DeleteWhenStopped)
            self.dock_animations[dock_widget] = animation

    def animate_dock_toggle(self, dock_widget, show_state_after_toggle):
        animation_duration = 150
        easing_curve = QEasingCurve.InOutQuad
        is_left_or_right_dock = self.dockWidgetArea(dock_widget) in [
            Qt.LeftDockWidgetArea,
            Qt.RightDockWidgetArea,
        ]

        # Determine the dimension to animate
        if dock_widget == self.sidebarDock and self.sidebar_content_widget:
            original_dimension = (
                self.sidebar_content_widget.maximumWidth()
                if not self.sidebar_content_widget._is_collapsed
                else self.sidebar_content_widget.minimumWidth()
            )
        else:
            default_size = 300 if is_left_or_right_dock else 200
            current_size = (
                dock_widget.width() if is_left_or_right_dock else dock_widget.height()
            )
            original_dimension = self.dock_original_sizes.get(
                dock_widget, current_size if current_size > 0 else default_size
            )

        # Stop any currently running animation on this dock
        if (
            dock_widget in self.dock_animations
            and self.dock_animations[dock_widget].state() == QPropertyAnimation.Running
        ):
            self.dock_animations[dock_widget].stop()

        if show_state_after_toggle:
            self._animate_dock_show(
                dock_widget,
                is_left_or_right_dock,
                original_dimension,
                animation_duration,
                easing_curve,
            )
        else:
            self._animate_dock_hide(
                dock_widget,
                is_left_or_right_dock,
                original_dimension,
                animation_duration,
                easing_curve,
            )

    @Slot(str)
    def switch_theme(self, theme_path):
        if theme_path != self.current_theme_path:
            self.current_theme_path = theme_path
            self.load_stylesheets()
            theme_name = "dark" if "dark" in theme_path else "light"
            if hasattr(self, "sidebar_content_widget"):
                self.sidebar_content_widget.update_icons(theme_name)
            if hasattr(self, "analysisPanel"):
                self.analysisPanel.update_theme_icons(theme_name)
            if hasattr(self, "custom_title_bar_widget"):
                self.custom_title_bar_widget.update_theme_icons(theme_name)
            if hasattr(self, "viewerPanel"):
                self.viewerPanel.update_theme(theme_name)
            if hasattr(self, "pythonTerminal"):
                self.pythonTerminal.set_theme(theme_name)

    @Slot()
    def refresh_all_gui_panels(self):
        self.connector.opticChanged.emit()

    @Slot()
    def new_system_action(self):
        self.connector.new_system()
        self._update_project_name_in_title_bar()
        print("Main Window: New System action triggered")

    @Slot()
    def open_system_action(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Optiland System",
            "",
            "Optiland JSON Files (*.json);;All Files (*)",
        )
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
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Optiland System As...",
            "",
            "Optiland JSON Files (*.json);;All Files (*)",
        )
        if filepath:
            if (
                not filepath.lower().endswith(".json")
                and "(*.json)" in _.split(";;")[0]
            ):
                filepath += ".json"
            self.connector.save_optic_to_file(filepath)
            self._update_project_name_in_title_bar()
            print(f"Main Window: Save System As action triggered - {filepath}")

    @Slot()
    def about_action(self):
        if not self.about_dialog:
            self.about_dialog = QDialog(self)
            self.about_dialog.setWindowTitle("About Optiland GUI")
            layout = QVBoxLayout(self.about_dialog)
            about_text = QLabel(
                "<p><b>Optiland GUI</b></p>"
                "<p>A modern interface for the Optiland optical simulation package.</p>"
                "<p>Version: 0.2.1 (Frameless Layout Refined)</p>"
                "<p>Built with PySide6.</p>"
                "<hr>"
                "<p><b>Icon Copyright Notice:</b></p>"
                "<p>Icons are provided under the MIT License.</p>"
                "<p>Copyright (c) 2020-2024 Paweł Kuna</p>"
            )
            about_text.setTextFormat(Qt.TextFormat.RichText)
            about_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(about_text)
            ok_button = QPushButton("OK")
            ok_button.clicked.connect(self.about_dialog.accept)
            layout.addWidget(ok_button, alignment=Qt.AlignmentFlag.AlignCenter)
            self.about_dialog.setLayout(layout)
            self.about_dialog.setMinimumSize(350, 220)
            self.about_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self.about_dialog.setWindowOpacity(0.0)
            self.about_dialog_animation = QPropertyAnimation(
                self.about_dialog, b"windowOpacity"
            )
            self.about_dialog_animation.setDuration(300)
            self.about_dialog_animation.setStartValue(0.0)
            self.about_dialog_animation.setEndValue(1.0)
            self.about_dialog_animation.setEasingCurve(QEasingCurve.InOutQuad)
            self.about_dialog_animation.start(
                QPropertyAnimation.DeletionPolicy.DeleteWhenStopped
            )
        self.about_dialog.exec()

    @Slot()
    def reset_windows_action(self):
        print(
            "Main Window: Reset Windows action triggered - resetting to "
            "revised default layout."
        )
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
        QMessageBox.information(
            self,
            "Layout Saved",
            f"The current window layout was saved to configuration - {target_slot}",
        )
        self.next_save_slot_index = 2 if target_slot == 1 else 1
        self.settings.setValue("Layouts/NextSaveSlot", self.next_save_slot_index)
        self.loadLayout1Action.setEnabled(
            self.settings.contains("Layouts/Config1Geometry")
        )
        self.loadLayout2Action.setEnabled(
            self.settings.contains("Layouts/Config2Geometry")
        )
        print(
            f"Layout saved to slot {target_slot}. Next save will be to slot "
            "{self.next_save_slot_index}."
        )

    def _load_layout_from_slot(self, slot_number):
        geometry_key = f"Layouts/Config{slot_number}Geometry"
        state_key = f"Layouts/Config{slot_number}State"
        if self.settings.contains(geometry_key) and self.settings.contains(state_key):
            window_geometry = self.settings.value(geometry_key)
            dock_toolbar_state = self.settings.value(state_key)
            if isinstance(window_geometry, QByteArray) and isinstance(
                dock_toolbar_state, QByteArray
            ):
                if not self.restoreGeometry(window_geometry):
                    print(
                        "Warning: Failed to restore window geometry from "
                        "slot {slot_number}."
                    )
                if not self.restoreState(dock_toolbar_state):
                    print(
                        "Warning: Failed to restore dock/toolbar state from "
                        "slot {slot_number}."
                    )
                QMessageBox.information(
                    self,
                    "Layout Loaded",
                    f"Layout from configuration - {slot_number} has been loaded.",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Load Error",
                    f"Invalid layout data found in configuration - {slot_number}.",
                )
        else:
            QMessageBox.information(
                self,
                "Load Layout",
                f"No layout saved in configuration - {slot_number}.",
            )

    # logic related to freely moving the main window
    def mousePressEvent(self, event):
        """Handle mouse press events for window dragging and resizing."""
        if event.button() == Qt.LeftButton:
            cursor_pos = event.position().toPoint()
            self.resize_area = self._get_resize_area(cursor_pos)

            if self.resize_area:
                self.is_resizing = True
            # Check for dragging the title bar
            elif self.custom_title_bar_widget:
                titlebar_rect = self.custom_title_bar_widget.rect()
                global_pos = self.custom_title_bar_widget.mapToGlobal(QPoint(0, 0))
                window_pos = self.mapFromGlobal(global_pos)
                titlebar_rect.moveTo(window_pos)

                if titlebar_rect.contains(cursor_pos) and not self.isMaximized():
                    self.is_moving = True
                    self.drag_position = (
                        event.globalPosition().toPoint()
                        - self.frameGeometry().topLeft()
                    )

            self.start_geometry = self.geometry()
            self.start_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        """Handle mouse move events for window dragging, resizing, and AeroSnap."""
        cursor_pos = event.position().toPoint()
        global_pos = event.globalPosition().toPoint()

        # Update cursor based on position
        self.updateCursorShape(cursor_pos)

        if self.is_resizing and not self.isMaximized():
            # Handle resizing
            diff = global_pos - self.start_pos
            new_geometry = QRect(self.start_geometry)

            if self.resize_area in ["top_left", "top", "top_right"]:
                new_geometry.setTop(self.start_geometry.top() + diff.y())
            if self.resize_area in ["bottom_left", "bottom", "bottom_right"]:
                new_geometry.setBottom(self.start_geometry.bottom() + diff.y())
            if self.resize_area in ["top_left", "left", "bottom_left"]:
                new_geometry.setLeft(self.start_geometry.left() + diff.x())
            if self.resize_area in ["top_right", "right", "bottom_right"]:
                new_geometry.setRight(self.start_geometry.right() + diff.x())

            # Enforce minimum size
            if (
                new_geometry.width() >= self.minimumWidth()
                and new_geometry.height() >= self.minimumHeight()
            ):
                self.setGeometry(new_geometry)

        elif self.is_moving and not self.isMaximized():
            # Handle window movement with AeroSnap
            self.move(global_pos - self.drag_position)

            # AeroSnap implementation
            screen = self.screen()
            screen_geometry = screen.availableGeometry()

            # Detect screen edges for snapping
            at_top_edge = global_pos.y() <= screen_geometry.top() + 5
            at_left_edge = global_pos.x() <= screen_geometry.left() + 5
            at_right_edge = global_pos.x() >= screen_geometry.right() - 5

            # Apply cursor feedback for AeroSnap zones
            if at_top_edge or at_left_edge or at_right_edge:
                self.setCursor(Qt.ArrowCursor)
                # Show right-half preview outline if needed

    def mouseReleaseEvent(self, event):
        """Handle mouse release events and implement AeroSnap actions."""
        if event.button() == Qt.LeftButton:
            # Implement AeroSnap functionality
            if self.is_moving and not self.isMaximized():
                screen = self.screen()
                screen_geometry = screen.availableGeometry()
                global_pos = event.globalPosition().toPoint()

                # Detect screen edges for snapping actions
                at_top_edge = global_pos.y() <= screen_geometry.top() + 5
                at_left_edge = global_pos.x() <= screen_geometry.left() + 5
                at_right_edge = global_pos.x() >= screen_geometry.right() - 5

                if at_top_edge:
                    # Maximize window
                    self.showMaximized()
                elif at_left_edge:
                    # Snap to left half
                    self.showNormal()
                    new_geometry = QRect(
                        screen_geometry.left(),
                        screen_geometry.top(),
                        screen_geometry.width() // 2,
                        screen_geometry.height(),
                    )
                    self.setGeometry(new_geometry)
                elif at_right_edge:
                    # Snap to right half
                    self.showNormal()
                    new_geometry = QRect(
                        screen_geometry.left() + screen_geometry.width() // 2,
                        screen_geometry.top(),
                        screen_geometry.width() // 2,
                        screen_geometry.height(),
                    )
                    self.setGeometry(new_geometry)

            # Reset states
            self.is_moving = False
            self.is_resizing = False
            self.setCursor(Qt.ArrowCursor)

    def _get_resize_area(self, pos: QPoint) -> str | None:
        """Determines which resize area the mouse cursor is in."""
        rect = self.rect()
        grip = self.grip_size

        if QRect(0, 0, grip, grip).contains(pos):
            return "top_left"
        if QRect(rect.width() - grip, 0, grip, grip).contains(pos):
            return "top_right"
        if QRect(0, rect.height() - grip, grip, grip).contains(pos):
            return "bottom_left"
        if QRect(rect.width() - grip, rect.height() - grip, grip, grip).contains(pos):
            return "bottom_right"
        if QRect(grip, 0, rect.width() - 2 * grip, grip).contains(pos):
            return "top"
        if QRect(0, grip, grip, rect.height() - 2 * grip).contains(pos):
            return "left"
        if QRect(
            rect.width() - grip, grip, rect.width(), rect.height() - 2 * grip
        ).contains(pos):
            return "right"
        if QRect(grip, rect.height() - grip, rect.width() - 2 * grip, grip).contains(
            pos
        ):
            return "bottom"

        return None

    def updateCursorShape(self, pos):
        """Update the cursor shape based on the mouse position."""
        if self.isMaximized():
            self.setCursor(Qt.ArrowCursor)
            return

        resize_area = self._get_resize_area(pos)

        if resize_area in ("top_left", "bottom_right"):
            self.setCursor(Qt.SizeFDiagCursor)
        elif resize_area in ("top_right", "bottom_left"):
            self.setCursor(Qt.SizeBDiagCursor)
        elif resize_area in ("top", "bottom"):
            self.setCursor(Qt.SizeVerCursor)
        elif resize_area in ("left", "right"):
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, event):
        """Handle key events for window management shortcuts."""
        # ctrl key + Left/Right/Up/Down for window snapping
        if event.modifiers() & Qt.ControlModifier:
            screen = self.screen()
            screen_geometry = screen.availableGeometry()

            if event.key() == Qt.Key_Left:
                # Snap to left half
                self.showNormal()
                self.setGeometry(
                    QRect(
                        screen_geometry.left(),
                        screen_geometry.top(),
                        screen_geometry.width() // 2,
                        screen_geometry.height(),
                    )
                )
            elif event.key() == Qt.Key_Right:
                # Snap to right half
                self.showNormal()
                self.setGeometry(
                    QRect(
                        screen_geometry.left() + screen_geometry.width() // 2,
                        screen_geometry.top(),
                        screen_geometry.width() // 2,
                        screen_geometry.height(),
                    )
                )
            elif event.key() == Qt.Key_Up:
                # Maximize
                self.showMaximized()
            elif event.key() == Qt.Key_Down:
                # Restore/minimize
                if self.isMaximized():
                    self.showNormal()
                else:
                    self.showMinimized()
        else:
            super().keyPressEvent(event)

    @Slot()
    def load_layout_1_slot(self):
        print("Loading layout from slot 1...")
        self._load_layout_from_slot(1)

    @Slot()
    def load_layout_2_slot(self):
        print("Loading layout from slot 2...")
        self._load_layout_from_slot(2)

    @Slot(str)
    def on_sidebar_menu_selected(self, button_name):
        dock_map = {
            "analysis": self.analysisPanelDock,
            "scripts": self.terminalDock,
            "design": self.lensEditorDock,
        }
        dock = dock_map.get(button_name)
        if dock:
            dock.show()
            dock.raise_()

    def closeEvent(self, event: QEvent):
        print("Main Window: Closing application.")
        self.pythonTerminal.shutdown_kernel()
        event.accept()

    @Slot()
    def show_settings_wip(self):
        """Shows a 'Work in Progress' message for the settings panel."""
        QMessageBox.information(
            self,
            "Work in Progress",
            "The settings panel is currently under development.",
        )

    # logic to load samples from the samples folder in optiland
    def _populate_gallery_menu(self, menu_bar: QMenuBar):
        """Creates the 'Gallery' menu by inspecting the samples package."""
        gallery_menu = menu_bar.addMenu("&Gallery")
        samples_menu = gallery_menu.addMenu("&Samples")

        systems_by_module = defaultdict(list)

        for _, obj_class in inspect.getmembers(optiland.samples, inspect.isclass):
            if issubclass(obj_class, Optic) and obj_class is not Optic:
                module_name = obj_class.__module__.split(".")[-1]
                systems_by_module[module_name].append(obj_class)

        if not systems_by_module:
            samples_menu.addAction("No samples found.").setEnabled(False)
            return

        for module_name, classes in sorted(systems_by_module.items()):
            submenu_name = module_name.replace("_", " ").title()
            submenu = samples_menu.addMenu(submenu_name)
            for optic_class in sorted(classes, key=lambda c: c.__name__):
                action_name = optic_class.__name__.replace("_", " ").title()
                action = QAction(action_name, self)
                action.triggered.connect(
                    lambda checked=False, cls=optic_class: self._load_sample_action(cls)
                )
                submenu.addAction(action)

    def _load_sample_action(self, optic_class: type[Optic]):
        """Instantiates and loads the selected sample class."""
        try:
            optic_instance = optic_class()
            self.connector.load_optic_from_object(optic_instance)
            print(f"Loaded sample: {optic_class.__name__}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Sample Load Error",
                f"Could not load sample '{optic_class.__name__}':\n{e}",
            )
