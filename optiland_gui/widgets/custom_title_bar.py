"""
Provides a custom title bar for the main application window.

This module defines `CustomTitleBar`, a QWidget that replaces the default
window title bar to allow for custom styling and additional widgets, such as
a main menu bar and project information label.

@author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

import webbrowser

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QToolButton,
    QWidget,
)


class CustomTitleBar(QWidget):
    """
    A custom title bar widget for the main window.

    This class creates a custom title bar with application title, a menu bar,
    project name label, and custom-styled minimize, maximize, and close buttons.
    It also handles window movement by dragging the title bar.

    Signals:
        minimize_requested: Emitted when the minimize button is clicked.
        maximize_restore_requested: Emitted when the maximize/restore button is clicked.
        close_requested: Emitted when the close button is clicked.

    Attributes:
        main_menu_bar (QMenuBar): The main menu bar instance to be displayed.
        title_label (QLabel): The label for the application title.
        project_label (QLabel): The label to display the current project name.
        minimize_button (QPushButton): The button to minimize the window.
        maximize_button (QPushButton): The button to maximize or restore the window.
        close_button (QPushButton): The button to close the window.
    """

    minimize_requested = Signal()
    maximize_restore_requested = Signal()
    close_requested = Signal()
    settings_requested = Signal()

    def __init__(self, main_menu_bar_instance: QMenuBar, parent=None):
        """
        Initializes the CustomTitleBar.

        Args:
            main_menu_bar_instance (QMenuBar): The QMenuBar to be embedded
            in the title bar.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.current_theme = "dark"
        self.setObjectName("CustomTitleBar")
        self.setFixedHeight(40)
        self.setAutoFillBackground(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 5, 0)
        layout.setSpacing(10)

        self.title_label = QLabel("Optiland")
        self.title_label.setObjectName("TitleBarOptilandLabel")
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.title_label)

        self.main_menu_bar = main_menu_bar_instance
        if self.main_menu_bar:
            self.main_menu_bar.setObjectName("TitleBarMenuBar")
            self.main_menu_bar.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred
            )
            layout.addWidget(self.main_menu_bar, 0, Qt.AlignmentFlag.AlignCenter)

        # certical separator
        separator = QFrame()
        separator.setObjectName("TitleBarSeparator")
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        # buttons (settings, GitHub, help, future: include copyright)
        btn_size = QSize(30, 30)

        self.settings_button = QToolButton()
        self.settings_button.setObjectName("TitleBarSettingsButton")
        self.settings_button.setFixedSize(btn_size)
        self.settings_button.setToolTip("Settings")
        self.settings_button.clicked.connect(self.settings_requested.emit)
        layout.addWidget(self.settings_button)
        layout.addSpacing(-10)

        self.github_button = QToolButton()
        self.github_button.setObjectName("TitleBarGitHubButton")
        self.github_button.setFixedSize(btn_size)
        self.github_button.setToolTip("Open GitHub Page")
        self.github_button.clicked.connect(self._open_github_url)
        layout.addWidget(self.github_button)
        layout.addSpacing(-10)

        self.help_button = QToolButton()
        self.help_button.setObjectName("TitleBarHelpButton")
        self.help_button.setFixedSize(btn_size)
        self.help_button.setToolTip("Open Documentation")
        self.help_button.clicked.connect(self._open_help_url)
        layout.addWidget(self.help_button)

        layout.addStretch(1)

        self.project_label = QLabel("Current Project: UnnamedProject.opds")
        self.project_label.setObjectName("TitleBarProjectLabel")
        self.project_label.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.project_label)

        layout.addSpacerItem(
            QSpacerItem(20, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        )

        btn_size = QSize(30, 30)
        self.minimize_button = QPushButton()
        self.minimize_button.setObjectName("TitleBarMinimizeButton")
        self.minimize_button.setFixedSize(btn_size)
        self.minimize_button.setToolTip("Minimize")
        self.minimize_button.clicked.connect(self.minimize_requested.emit)
        layout.addWidget(self.minimize_button)

        self.maximize_button = QPushButton()
        self.maximize_button.setObjectName("TitleBarMaximizeButton")
        self.maximize_button.setFixedSize(btn_size)
        self.maximize_button.setCheckable(True)
        self.maximize_button.setToolTip("Maximize")
        self.maximize_button.clicked.connect(self.maximize_restore_requested.emit)
        layout.addWidget(self.maximize_button)

        self.close_button = QPushButton()
        self.close_button.setObjectName("TitleBarCloseButton")
        self.close_button.setFixedSize(btn_size)
        self.close_button.setToolTip("Close")
        self.close_button.clicked.connect(self.close_requested.emit)
        layout.addWidget(self.close_button)

        self.update_theme_icons()

        self._mouse_press_pos = None
        self._mouse_move_offset = None

    def update_theme_icons(self, theme="dark"):
        """
        Updates the icons for the title bar buttons based on the selected theme.

        Args:
            theme (str, optional): The theme name ('dark' or 'light').
            Defaults to "dark".
        """
        self.current_theme = theme
        self.minimize_button.setIcon(QIcon(f":/icons/{theme}/minimize.svg"))
        self.close_button.setIcon(QIcon(f":/icons/{theme}/close.svg"))
        self.maximize_button.setIcon(QIcon(f":/icons/{theme}/maximize_restore.svg"))
        self.settings_button.setIcon(QIcon(f":/icons/{theme}/settings.svg"))
        self.github_button.setIcon(QIcon(f":/icons/{theme}/brand_github.svg"))
        self.help_button.setIcon(QIcon(f":/icons/{theme}/help.svg"))

    def set_project_name(self, name: str):
        """
        Sets the display text for the current project name.

        Args:
            name (str): The name of the project. If empty, a default name is used.
        """
        if not name:
            name = "UnnamedProject.opds"
        self.project_label.setText(f"Current Project: {name}")

    def update_maximize_button_state(self, is_maximized: bool):
        """
        Updates the visual state and tooltip of the maximize/restore button.

        Args:
            is_maximized (bool): True if the window is maximized, False otherwise.
        """
        self.maximize_button.setChecked(is_maximized)
        self.maximize_button.setToolTip("Restore" if is_maximized else "Maximize")

    def mousePressEvent(self, event):
        """
        Handles mouse press events to initiate window dragging.

        Args:
            event (QMouseEvent): The mouse press event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_move_offset = (
                event.globalPosition().toPoint()
                - self.window().frameGeometry().topLeft()
            )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Handles mouse move events to drag the window.

        Args:
            event (QMouseEvent): The mouse move event.
        """
        if (
            self._mouse_move_offset is not None
            and event.buttons() == Qt.MouseButton.LeftButton
        ):
            new_window_pos = event.globalPosition().toPoint() - self._mouse_move_offset
            self.window().move(new_window_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Handles mouse release events to stop window dragging.

        Args:
            event (QMouseEvent): The mouse release event.
        """
        self._mouse_move_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """
        Handles double-click on title bar to maximize/restore window.

        Args:
            event (QMouseEvent): The mouse double-click event.
        """
        if event.button() == Qt.LeftButton:
            self.maximize_restore_requested.emit()

    def _open_github_url(self):
        """Opens the Optiland GitHub repository URL in a web browser."""
        url = "https://github.com/HarrisonKramer/optiland"
        webbrowser.open(url)

    def _open_help_url(self):
        """Opens the Optiland documentation URL in a web browser."""
        url = "https://optiland.readthedocs.io/en/latest/index.html"
        webbrowser.open(url)
