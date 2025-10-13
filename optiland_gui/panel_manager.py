"""
Manages the creation, layout, and lifecycle of dockable panels.

This module provides the `PanelManager` class, which is responsible for
instantiating, configuring, and arranging all the QDockWidget panels used in
the Optiland GUI. This decouples the main window from the specific details of
each panel.

Author: Jules, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

from .analysis_panel import AnalysisPanel
from .lens_editor import LensEditor
from .system_properties_panel import SystemPropertiesPanel
from .viewer_panel import ViewerPanel
from .widgets.custom_dock_widget import CustomDockWidget
from .widgets.python_terminal import PythonTerminalWidget
from .widgets.sidebar import SidebarWidget

if TYPE_CHECKING:
    from .optiland_connector import OptilandConnector


class PanelManager:
    """Manages the instantiation and layout of all GUI panels."""

    def __init__(self, main_window: QMainWindow, connector: OptilandConnector):
        self.main_window = main_window
        self.connector = connector
        self.all_docks = []
        self.sidebar = None

    def create_all_panels(self, parent_for_iface: QWidget):
        """Creates instances of all panels and their containing docks."""
        # Sidebar
        self.sidebar_content_widget = SidebarWidget(self.main_window)
        self.sidebar = self._create_dock(
            widget=self.sidebar_content_widget,
            name="SidebarDockWidget",
            title="NavigationSidebar",
            features=(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable),
            allowed_areas=Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea,
            use_custom_title_bar=False,
        )
        self.sidebar.setTitleBarWidget(QWidget())

        # Main Panels
        self.viewer_panel = ViewerPanel(self.connector)
        self.viewer_dock = self._create_dock(
            self.viewer_panel, "ViewerDock", "System Viewer"
        )

        self.lens_editor = LensEditor(self.connector)
        self.lens_editor_dock = self._create_dock(
            self.lens_editor, "LensEditorDock", "Lens Data Editor"
        )

        self.system_properties = SystemPropertiesPanel(self.connector)
        self.system_properties_dock = self._create_dock(
            self.system_properties, "SystemPropertiesDock", "System Properties"
        )

        self.analysis_panel = AnalysisPanel(self.connector)
        self.analysis_dock = self._create_dock(
            self.analysis_panel, "AnalysisPanelDock", "Analysis"
        )

        # Terminal
        initial_theme = "dark"  # TODO: Get this from settings or main_window
        self.python_terminal = PythonTerminalWidget(
            parent_for_iface,
            custom_variables={
                "connector": self.connector,
                "iface": parent_for_iface.iface,
            },
            theme=initial_theme,
        )
        self.terminal_dock = self._create_dock(
            self.python_terminal, "TerminalDock", "Scripts Terminal"
        )

        self.all_docks = [
            self.sidebar,
            self.viewer_dock,
            self.lens_editor_dock,
            self.system_properties_dock,
            self.analysis_dock,
            self.terminal_dock,
        ]

    def _create_dock(
        self,
        widget: QWidget,
        name: str,
        title: str,
        features=None,
        allowed_areas=Qt.AllDockWidgetAreas,
        use_custom_title_bar=True,
    ) -> QDockWidget:
        """Factory method for creating and configuring a QDockWidget."""
        if use_custom_title_bar:
            dock = CustomDockWidget(title, self.main_window)
            dock.setWidget(widget)
        else:
            dock = QDockWidget(title, self.main_window)
            dock.setWidget(widget)

        dock.setObjectName(name)
        if features is not None:
            dock.setFeatures(features)
        else:
            dock.setFeatures(
                QDockWidget.DockWidgetClosable
                | QDockWidget.DockWidgetMovable
                | QDockWidget.DockWidgetFloatable
            )
        dock.setAllowedAreas(allowed_areas)
        return dock

    def setup_default_layout(self):
        """Arranges the dock widgets into the default layout."""
        for dock in self.all_docks:
            if dock:
                dock.setFloating(False)
                self.main_window.addDockWidget(Qt.LeftDockWidgetArea, dock)
                dock.show()

        # Layout logic
        self.main_window.addDockWidget(Qt.LeftDockWidgetArea, self.sidebar)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.lens_editor_dock)
        self.main_window.splitDockWidget(
            self.lens_editor_dock, self.analysis_dock, Qt.Horizontal
        )
        self.main_window.splitDockWidget(
            self.lens_editor_dock, self.viewer_dock, Qt.Vertical
        )
        self.main_window.splitDockWidget(
            self.analysis_dock, self.terminal_dock, Qt.Vertical
        )
        self.main_window.tabifyDockWidget(
            self.analysis_dock, self.system_properties_dock
        )

        # Ensure panels are raised
        for dock in reversed(self.all_docks):
            if dock:
                dock.raise_()

    def get_all_docks(self) -> list[QDockWidget]:
        """Returns a list of all managed dock widgets."""
        return self.all_docks

    def connect_signals(self):
        """Connect signals for the managed panels."""
        self.sidebar_content_widget.menuSelected.connect(self.on_sidebar_menu_selected)
        self.python_terminal.commandExecuted.connect(self.connector.opticChanged.emit)

    def on_sidebar_menu_selected(self, button_name: str):
        """Shows and raises the corresponding dock when a sidebar button is clicked."""
        dock_map = {
            "analysis": self.analysis_dock,
            "scripts": self.terminal_dock,
            "design": self.lens_editor_dock,
        }
        dock = dock_map.get(button_name)
        if dock:
            dock.show()
            dock.raise_()

    def update_theme(self, theme_name: str):
        """Propagates theme changes to all relevant panels."""
        self.sidebar_content_widget.update_icons(theme_name)
        # Call the full theme update method on panels with plots
        self.analysis_panel.update_theme(theme_name)
        self.viewer_panel.update_theme(theme_name)
        self.python_terminal.set_theme(theme_name)
