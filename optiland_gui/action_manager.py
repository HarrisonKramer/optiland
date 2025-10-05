"""
Manages the creation and handling of QActions for the Optiland GUI.

This module provides the `ActionManager` class, which is responsible for
instantiating and configuring all the QAction objects used in the application's
menus and toolbars. This separates the action definitions from the main window logic.

Author: Jules, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QAction, QActionGroup, QKeySequence

from .config import THEME_DARK_PATH, THEME_LIGHT_PATH

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow


class ActionManager:
    """Creates and manages all QAction objects for the application."""

    def __init__(self, main_window: QMainWindow, connector):
        self.main_window = main_window
        self.connector = connector
        self.actions = {}

    def create_all_actions(self):
        """Creates all actions and stores them in the `actions` dictionary."""
        self._create_file_actions()
        self._create_edit_actions()
        self._create_view_actions()
        self._create_layout_actions()
        self._create_theme_actions()
        self._create_help_actions()

    def _create_action(
        self, name, text, shortcut=None, triggered=None, tooltip=None, checkable=False
    ):
        """Factory method for creating a QAction."""
        action = QAction(text, self.main_window, checkable=checkable)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if triggered:
            action.triggered.connect(triggered)
        if tooltip:
            action.setToolTip(tooltip)
        self.actions[name] = action
        return action

    def _create_file_actions(self):
        self._create_action(
            "new", "&New System", QKeySequence.New, self.main_window.new_system_action
        )
        self._create_action(
            "open",
            "&Open System...",
            QKeySequence.Open,
            self.main_window.open_system_action,
        )
        self._create_action(
            "save",
            "&Save System",
            QKeySequence.Save,
            self.main_window.save_system_action,
        )
        self._create_action(
            "save_as",
            "Save System &As...",
            QKeySequence.SaveAs,
            self.main_window.save_system_as_action,
        )
        self._create_action("exit", "E&xit", "Ctrl+Q", self.main_window.close)

    def _create_edit_actions(self):
        undo = self._create_action(
            "undo", "&Undo", QKeySequence.Undo, self.connector.undo
        )
        redo = self._create_action(
            "redo", "&Redo", QKeySequence.Redo, self.connector.redo
        )
        undo.setEnabled(False)
        redo.setEnabled(False)
        self.connector.undoStackAvailabilityChanged.connect(undo.setEnabled)
        self.connector.redoStackAvailabilityChanged.connect(redo.setEnabled)

    def _create_view_actions(self):
        self._create_action(
            "dock_all",
            "Dock All Windows",
            triggered=self.main_window.reset_windows_action,
        )
        self._create_action(
            "reset_layout",
            "Reset Window Layout",
            triggered=self.main_window.reset_windows_action,
        )

    def _create_layout_actions(self):
        settings = self.main_window.settings
        load1 = self._create_action(
            "load_layout_1",
            "1",
            triggered=self.main_window.load_layout_1_slot,
            tooltip="Load Layout from Slot 1",
        )
        load2 = self._create_action(
            "load_layout_2",
            "2",
            triggered=self.main_window.load_layout_2_slot,
            tooltip="Load Layout from Slot 2",
        )
        self._create_action(
            "save_layout",
            "Save Current Layout",
            triggered=self.main_window.save_layout_slot,
            tooltip="Save current window layout to next available slot (1 or 2)",
        )
        load1.setEnabled(settings.contains("Layouts/Config1Geometry"))
        load2.setEnabled(settings.contains("Layouts/Config2Geometry"))

    def _create_theme_actions(self):
        group = QActionGroup(self.main_window)
        group.setExclusive(True)
        dark_action = self._create_action(
            "dark_theme",
            "Dark Theme",
            checkable=True,
            triggered=lambda: self.main_window.switch_theme(THEME_DARK_PATH),
        )
        light_action = self._create_action(
            "light_theme",
            "Light Theme",
            checkable=True,
            triggered=lambda: self.main_window.switch_theme(THEME_LIGHT_PATH),
        )
        group.addAction(dark_action)
        group.addAction(light_action)
        self.actions["theme_group"] = group

    def _create_help_actions(self):
        self._create_action(
            "about", "&About Optiland GUI", triggered=self.main_window.about_action
        )

    def get_action(self, name: str) -> QAction:
        """Retrieves a created action by its name."""
        return self.actions.get(name)

    def get_actions(self, *names: str) -> list[QAction]:
        """Retrieves multiple actions by their names."""
        return [self.actions.get(name) for name in names]
