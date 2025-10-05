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
    QPropertyAnimation,
    QSettings,
    Qt,
    Slot,
)
from PySide6.QtGui import QAction, QResizeEvent
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import optiland.samples
from optiland.optic import Optic

from . import gui_plot_utils
from .action_manager import ActionManager
from .config import (
    APPLICATION_NAME,
    ORGANIZATION_NAME,
    SIDEBAR_QSS_PATH,
    THEME_DARK_PATH,
)
from .optiland_connector import OptilandConnector
from .panel_manager import PanelManager

# from .optimization_panel import OptimizationPanel # we will support this later on
from .widgets.custom_title_bar import CustomTitleBar
from .widgets.frameless_window import FramelessWindow
from .widgets.sidebar import (
    SIDEBAR_MAX_WIDTH,
    SIDEBAR_MIN_WIDTH,
)

try:
    from .resources import resources_rc  # noqa: F401
except ImportError as e:
    print(f"Warning (main_window.py): Could not import resources_rc.py: {e}")


class MainWindow(FramelessWindow):
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
            return self._win.panel_manager.analysis_panel

        def get_lens_editor(self):
            """Returns the LensEditor widget instance.

            Returns:
                LensEditor: The lens data editor widget.
            """
            return self._win.panel_manager.lens_editor

        def get_viewer_panel(self):
            """Returns the ViewerPanel widget instance.

            Returns:
                ViewerPanel: The 2D/3D viewer panel widget.
            """
            return self._win.panel_manager.viewer_panel

        def show_lens_editor(self):
            """Brings the Lens Data Editor dock widget to the front."""
            self._win.panel_manager.lens_editor_dock.show()
            self._win.panel_manager.lens_editor_dock.raise_()

        def show_analysis_panel(self):
            """Brings the Analysis Panel dock widget to the front."""
            dock = self._win.panel_manager.analysis_dock
            dock.show()
            dock.raise_()
            # Also ensure its tab is selected
            parent_tab_widget = dock.parentWidget()
            if isinstance(parent_tab_widget, QTabWidget):
                parent_tab_widget.setCurrentWidget(dock)

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
        self.panel_manager.create_all_panels(self)
        self.action_manager.create_all_actions()
        self._setup_menus_and_toolbars()
        self._setup_layout()

    def _configure_window(self):
        """Sets up the main window's flags, title, and geometry."""
        self.setWindowTitle("Optiland GUI")
        self.setGeometry(100, 100, 1600, 900)

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
        self.panel_manager = PanelManager(self, self.connector)
        self.action_manager = ActionManager(self, self.connector)
        self.dock_animations = {}
        self.dock_original_sizes = {}
        self.about_dialog = None

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
        dark_theme_action = self.action_manager.get_action("dark_theme")
        light_theme_action = self.action_manager.get_action("light_theme")
        if dark_theme_action and light_theme_action:
            dark_theme_action.setChecked(self.current_theme_path == THEME_DARK_PATH)
            light_theme_action.setChecked(self.current_theme_path != THEME_DARK_PATH)

        self._connect_dock_animations()
        self.panel_manager.connect_signals()

        self.connector.opticLoaded.connect(self._update_project_name_in_title_bar)
        self.connector.opticChanged.connect(self._update_project_name_in_title_bar)
        self.connector.modifiedStateChanged.connect(
            self._update_project_name_in_title_bar
        )

        self._initial_narrow_check_done = False
        self._update_project_name_in_title_bar()

    def _apply_revised_default_dock_layout(self):
        """Applies the default docking layout to the main window.

        This function arranges the dock widgets in a predefined layout, splitting
        and tabbing them to create a functional and organized user interface.
        This is called on first launch and when resetting the layout.
        """
        self.panel_manager.setup_default_layout()

        # Initial plot/render
        viewer_panel = self.panel_manager.viewer_panel
        if viewer_panel.viewer2D and hasattr(viewer_panel.viewer2D, "plot_optic"):
            viewer_panel.viewer2D.plot_optic()
        if viewer_panel.viewer3D and hasattr(viewer_panel.viewer3D, "render_optic"):
            viewer_panel.viewer3D.render_optic()

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
            sidebar_widget = self.panel_manager.sidebar_content_widget
            sidebar_dock = self.panel_manager.sidebar
            if hasattr(self, "panel_manager") and sidebar_widget and sidebar_dock:
                if self.width() < (SIDEBAR_MAX_WIDTH + 300):
                    sidebar_widget.force_set_collapse_state(True)
                    if sidebar_dock.width() > SIDEBAR_MIN_WIDTH:
                        self.resizeDocks(
                            [sidebar_dock], [SIDEBAR_MIN_WIDTH], Qt.Horizontal
                        )
                else:
                    sidebar_widget.force_set_collapse_state(False)
                    if sidebar_dock.width() < 150:
                        self.resizeDocks([sidebar_dock], [150], Qt.Horizontal)
            self._initial_narrow_check_done = True

        dark_theme_action = self.action_manager.get_action("dark_theme")
        light_theme_action = self.action_manager.get_action("light_theme")
        if dark_theme_action and light_theme_action:
            is_dark = self.current_theme_path == THEME_DARK_PATH
            dark_theme_action.setChecked(is_dark)
            light_theme_action.setChecked(not is_dark)

    def _connect_dock_animations(self):
        """Connects dock widget view actions to an animation handler.

        This method disconnects the default `triggered` signal from each dock's
        toggle view action and reconnects it to a custom slot that provides a
        fade-in/out animation for a smoother user experience.
        """
        if not hasattr(self, "panel_manager"):
            return
        for dock_widget_ref in self.panel_manager.get_all_docks():
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

    def _populate_main_menu_bar(self, menu_bar: QMenuBar):
        """Populates the main menu bar with actions and sub-menus.

        Args:
            menu_bar: The QMenuBar instance to populate.
        """
        am = self.action_manager
        file_menu = menu_bar.addMenu("&File")
        file_menu.addActions(am.get_actions("new", "open", "save", "save_as"))
        file_menu.addSeparator()
        file_menu.addAction(am.get_action("exit"))

        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addActions(am.get_actions("undo", "redo"))

        view_menu = menu_bar.addMenu("&View")
        view_menu.addActions(am.get_actions("dock_all", "reset_layout"))
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("&Theme")
        theme_menu.addActions(am.get_actions("dark_theme", "light_theme"))
        view_menu.addSeparator()

        # Add toggle actions for all managed docks
        if hasattr(self, "panel_manager"):
            for dock in self.panel_manager.get_all_docks():
                if dock and dock.toggleViewAction():
                    # Create a friendlier name for the menu
                    action_text = f"Toggle {dock.windowTitle()}"
                    if "Sidebar" in dock.objectName():
                        action_text = "Toggle Navigation Sidebar"
                    dock.toggleViewAction().setText(action_text)
                    view_menu.addAction(dock.toggleViewAction())

        self._populate_gallery_menu(menu_bar)

        menu_bar.addMenu("&Run")
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(am.get_action("about"))

    def _populate_quick_actions_toolbar(self, toolbar: QToolBar):
        """Populates the quick actions toolbar with common actions.

        Args:
            toolbar: The QToolBar instance to populate.
        """
        am = self.action_manager
        toolbar.addActions(am.get_actions("new", "open", "save"))
        toolbar.addSeparator()
        toolbar.addActions(
            am.get_actions("load_layout_1", "load_layout_2", "save_layout")
        )
        toolbar.addSeparator()
        toolbar.addActions(am.get_actions("dock_all", "reset_layout"))

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

        dark_theme_action = self.action_manager.get_action("dark_theme")
        light_theme_action = self.action_manager.get_action("light_theme")
        if dark_theme_action and light_theme_action:
            is_dark = self.current_theme_path == THEME_DARK_PATH
            dark_theme_action.setChecked(is_dark)
            light_theme_action.setChecked(not is_dark)

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
        if (
            hasattr(self, "panel_manager")
            and dock_widget == self.panel_manager.sidebar
            and self.panel_manager.sidebar_content_widget
        ):
            original_dimension = (
                self.panel_manager.sidebar_content_widget.maximumWidth()
                if not self.panel_manager.sidebar_content_widget._is_collapsed
                else self.panel_manager.sidebar_content_widget.minimumWidth()
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

            if hasattr(self, "panel_manager"):
                self.panel_manager.update_theme(theme_name)
            if hasattr(self, "custom_title_bar_widget"):
                self.custom_title_bar_widget.update_theme_icons(theme_name)

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
        if hasattr(self, "panel_manager"):
            for dock in self.panel_manager.get_all_docks():
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
        load_layout_1 = self.action_manager.get_action("load_layout_1")
        if load_layout_1:
            load_layout_1.setEnabled(self.settings.contains("Layouts/Config1Geometry"))

        load_layout_2 = self.action_manager.get_action("load_layout_2")
        if load_layout_2:
            load_layout_2.setEnabled(self.settings.contains("Layouts/Config2Geometry"))
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
        if hasattr(self, "panel_manager") and self.panel_manager.python_terminal:
            self.panel_manager.python_terminal.shutdown_kernel()
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
