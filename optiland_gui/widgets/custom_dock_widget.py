"""
Provides a QDockWidget with a custom title bar.

Author: Jules, 2025
"""

from __future__ import annotations

from PySide6.QtWidgets import QDockWidget, QWidget

from .custom_dock_title_bar import CustomDockTitleBar


class CustomDockWidget(QDockWidget):
    """A QDockWidget that uses a custom title bar by default."""

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(title, parent)
        self.setTitleBarWidget(CustomDockTitleBar(self, title))

    def setWidget(self, widget: QWidget):
        """Sets the content widget and ensures the title bar is not replaced."""
        # Keep our custom title bar
        current_title_bar = self.titleBarWidget()
        super().setWidget(widget)
        if not isinstance(self.titleBarWidget(), CustomDockTitleBar):
            self.setTitleBarWidget(current_title_bar)
