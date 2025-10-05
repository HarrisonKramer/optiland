"""
Provides a custom title bar for QDockWidgets.

Author: Manuel Fragata Mendes, 2025
Refactored by: Jules, 2025
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QStyleOption,
    QWidget,
)


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
