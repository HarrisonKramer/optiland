# optiland_gui/widgets/custom_title_bar.py
from PySide6.QtCore import Qt, QPoint, Signal, QSize
from PySide6.QtGui import QIcon  # For icons on buttons
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
)


class CustomTitleBar(QWidget):
    """
    A custom title bar widget that includes the application title,
    a menu bar, a project label, and window control buttons.
    """

    # Signals to be connected to MainWindow's slots
    minimize_requested = Signal()
    maximize_restore_requested = Signal()
    close_requested = Signal()

    def __init__(self, main_menu_bar_instance: QMenuBar, parent=None):
        super().__init__(parent)
        self.setObjectName("CustomTitleBar")
        # Adjust height to fit all elements comfortably, e.g., 35-45px
        self.setFixedHeight(40)
        self.setAutoFillBackground(True)  # Allows styling with QSS background-color

        layout = QHBoxLayout(self)
        # Adjust margins: Left, Top, Right, Bottom
        layout.setContentsMargins(10, 0, 5, 0)
        layout.setSpacing(10)  # Spacing between elements in the title bar

        # 1. Optiland Application Title
        self.title_label = QLabel("Optiland")
        self.title_label.setObjectName("TitleBarOptilandLabel")
        # Allow it to take fixed space, vertical preferred
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.title_label)

        # 2. Main Menu Bar (passed from MainWindow)
        self.main_menu_bar = main_menu_bar_instance
        if self.main_menu_bar:
            self.main_menu_bar.setObjectName("TitleBarMenuBar")
            # Important: Set size policy for the menu bar
            self.main_menu_bar.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred
            )
            layout.addWidget(self.main_menu_bar)

        # Spacer to push project label and window controls to the right
        layout.addStretch(1)

        # 3. Current Project Label
        self.project_label = QLabel(
            "Current Project: UnnamedProject.opds"
        )  # Placeholder text
        self.project_label.setObjectName("TitleBarProjectLabel")
        # Allow it to expand but not excessively, vertical preferred
        self.project_label.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.project_label)

        # Optional: Add some fixed spacing before window controls if stretch isn't enough
        layout.addSpacerItem(
            QSpacerItem(20, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        )

        # 4. Window Control Buttons (using text for now, icons are better)
        # It's highly recommended to use SVG icons for these buttons.
        # For simplicity here, I'm using text.
        btn_size = QSize(30, 30)  # Fixed size for control buttons

        self.minimize_button = QPushButton("_")
        self.minimize_button.setObjectName("TitleBarMinimizeButton")
        self.minimize_button.setFixedSize(btn_size)
        self.minimize_button.setToolTip("Minimize")
        self.minimize_button.clicked.connect(self.minimize_requested.emit)
        layout.addWidget(self.minimize_button)

        self.maximize_button = QPushButton(
            "[]"
        )  # Placeholder for maximize/restore icon
        self.maximize_button.setObjectName("TitleBarMaximizeButton")
        self.maximize_button.setFixedSize(btn_size)
        self.maximize_button.setCheckable(True)  # To reflect maximized state
        self.maximize_button.setToolTip("Maximize")
        self.maximize_button.clicked.connect(self.maximize_restore_requested.emit)
        layout.addWidget(self.maximize_button)

        self.close_button = QPushButton("X")
        self.close_button.setObjectName("TitleBarCloseButton")
        self.close_button.setFixedSize(btn_size)
        self.close_button.setToolTip("Close")
        self.close_button.clicked.connect(self.close_requested.emit)
        layout.addWidget(self.close_button)

        # For window dragging
        self._mouse_press_pos = None
        self._mouse_move_offset = (
            None  # Store offset from window top-left to click point
        )

    def set_project_name(self, name: str):
        """Updates the project name label."""
        if not name:
            name = "UnnamedProject.opds"  # Default if empty
        self.project_label.setText(f"Current Project: {name}")

    def update_maximize_button_state(self, is_maximized: bool):
        """Updates the appearance of the maximize/restore button."""
        self.maximize_button.setChecked(is_maximized)
        # Ideally, you'd switch icons here. For text:
        self.maximize_button.setText("[-]" if is_maximized else "[]")
        self.maximize_button.setToolTip("Restore" if is_maximized else "Maximize")

    # --- Mouse events for dragging the frameless window ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Store the offset of the click from the window's top-left corner
            self._mouse_move_offset = (
                event.globalPosition().toPoint()
                - self.window().frameGeometry().topLeft()
            )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._mouse_move_offset is not None
            and event.buttons() == Qt.MouseButton.LeftButton
        ):
            # Move the window by maintaining the initial click offset
            new_window_pos = event.globalPosition().toPoint() - self._mouse_move_offset
            self.window().move(new_window_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._mouse_move_offset = None
        super().mouseReleaseEvent(event)
