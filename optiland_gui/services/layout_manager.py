"""Layout preset manager for the Optiland GUI.

Provides :class:`LayoutManager` which saves and restores named panel layouts
using :meth:`QMainWindow.saveState` / :meth:`QMainWindow.restoreState`.

Built-in presets (see :data:`BUILTIN_PRESETS`) are shipped with the app and
cannot be deleted, but can be overridden by saving a user preset with the
same name.  User presets are stored in ``QSettings`` under
``"layouts/user/<name>"``.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QByteArray, QSettings
from PySide6.QtWidgets import QInputDialog, QMessageBox

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow

#: Names of presets that ship with the application.
BUILTIN_PRESET_NAMES: tuple[str, ...] = (
    "Standard",
    "Optimization",
    "Analysis",
    "Simple Visualization",
)

_SETTINGS_PREFIX = "layouts/user/"
_MAX_USER_PRESETS_IN_MENU = 10


class LayoutManager:
    """Manages named window-layout presets for a :class:`QMainWindow`.

    Args:
        main_window: The application's main window.
        settings: The :class:`QSettings` instance for persistent storage.
    """

    def __init__(self, main_window: QMainWindow, settings: QSettings) -> None:
        self._win = main_window
        self._settings = settings

    # ------------------------------------------------------------------
    # Built-in preset application
    # ------------------------------------------------------------------

    def apply_builtin(self, name: str) -> None:
        """Apply a built-in preset by *name*.

        Each built-in preset is implemented by showing/hiding specific dock
        widgets.  The state is applied without relying on a serialised blob so
        it always works even on first launch.

        Args:
            name: One of ``BUILTIN_PRESET_NAMES``.
        """
        if name == "Standard":
            self._set_visibility(
                show=["viewer", "lens_editor", "system_properties", "analysis"],
                hide=["python_terminal", "optimization"],
            )
        elif name == "Optimization":
            self._set_visibility(
                show=["lens_editor", "viewer", "optimization"],
                hide=["analysis", "python_terminal", "system_properties"],
            )
        elif name == "Analysis":
            self._set_visibility(
                show=["lens_editor", "viewer", "analysis"],
                hide=["optimization", "python_terminal", "system_properties"],
            )
        elif name == "Simple Visualization":
            self._set_visibility(
                show=["viewer"],
                hide=[
                    "lens_editor",
                    "analysis",
                    "optimization",
                    "python_terminal",
                    "system_properties",
                ],
            )

    def _set_visibility(self, show: list[str], hide: list[str]) -> None:
        pm = self._win.panel_manager
        _dock_map = {
            "viewer": getattr(pm, "viewer_dock", None),
            "lens_editor": getattr(pm, "lens_editor_dock", None),
            "analysis": getattr(pm, "analysis_dock", None),
            "optimization": getattr(pm, "optimization_dock", None),
            "system_properties": getattr(pm, "system_properties_dock", None),
            "python_terminal": getattr(pm, "python_terminal_dock", None),
        }
        for key in show:
            dock = _dock_map.get(key)
            if dock:
                dock.show()
                dock.raise_()
        for key in hide:
            dock = _dock_map.get(key)
            if dock:
                dock.hide()

    # ------------------------------------------------------------------
    # User preset CRUD
    # ------------------------------------------------------------------

    def save_current_as(self) -> str | None:
        """Prompt for a name and save the current layout as a user preset.

        Returns:
            The name under which the preset was saved, or ``None`` if
            the user cancelled the dialog.
        """
        name, ok = QInputDialog.getText(
            self._win,
            "Save Layout",
            "Layout name:",
        )
        if not ok or not name.strip():
            return None
        name = name.strip()
        state = self._win.saveState()
        geometry = self._win.saveGeometry()
        self._settings.setValue(f"{_SETTINGS_PREFIX}{name}/state", state)
        self._settings.setValue(f"{_SETTINGS_PREFIX}{name}/geometry", geometry)
        return name

    def load_user(self, name: str) -> bool:
        """Restore the layout saved under *name*.

        Args:
            name: The preset name.

        Returns:
            ``True`` on success, ``False`` if no such preset exists or the
            restore failed.
        """
        state_key = f"{_SETTINGS_PREFIX}{name}/state"
        geo_key = f"{_SETTINGS_PREFIX}{name}/geometry"
        if not self._settings.contains(state_key):
            QMessageBox.information(
                self._win,
                "Layout Not Found",
                f'No saved layout named "{name}".',
            )
            return False
        state = self._settings.value(state_key)
        geometry = self._settings.value(geo_key)
        if isinstance(state, QByteArray):
            self._win.restoreState(state)
        if isinstance(geometry, QByteArray):
            self._win.restoreGeometry(geometry)
        return True

    def delete_user(self, name: str) -> None:
        """Remove a user preset.

        Args:
            name: The preset name to delete.
        """
        self._settings.remove(f"{_SETTINGS_PREFIX}{name}")

    def user_preset_names(self) -> list[str]:
        """Return a sorted list of all saved user-preset names.

        Returns:
            List of preset name strings.
        """
        self._settings.beginGroup("layouts/user")
        keys = self._settings.childGroups()
        self._settings.endGroup()
        return sorted(keys)

    def show_manage_dialog(self) -> None:
        """Open a simple dialog to delete user presets."""
        from PySide6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QLabel,
            QListWidget,
            QPushButton,
            QVBoxLayout,
        )

        names = self.user_preset_names()
        dialog = QDialog(self._win)
        dialog.setWindowTitle("Manage Layouts")
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Saved user layouts:"))
        list_widget = QListWidget()
        list_widget.addItems(names)
        layout.addWidget(list_widget)

        def _delete() -> None:
            item = list_widget.currentItem()
            if not item:
                return
            name = item.text()
            reply = QMessageBox.question(
                dialog,
                "Delete Layout",
                f'Delete layout "{name}"?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.delete_user(name)
                list_widget.takeItem(list_widget.row(item))

        del_btn = QPushButton("Delete Selected")
        del_btn.clicked.connect(_delete)
        layout.addWidget(del_btn)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dialog.reject)
        layout.addWidget(bb)
        dialog.exec()
