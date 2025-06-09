# optiland_gui/widgets/custom_title_bar.py
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QWidget,
)


class CustomTitleBar(QWidget):
    minimize_requested = Signal()
    maximize_restore_requested = Signal()
    close_requested = Signal()

    def __init__(self, main_menu_bar_instance: QMenuBar, parent=None):
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

        self.minimize_button = QPushButton("_")
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
        self.current_theme = theme
        self.close_button.setIcon(QIcon(f":/icons/{theme}/close.svg"))
        self.maximize_button.setIcon(QIcon(f":/icons/{theme}/maximize_restore.svg"))

    def set_project_name(self, name: str):
        if not name:
            name = "UnnamedProject.opds"
        self.project_label.setText(f"Current Project: {name}")

    def update_maximize_button_state(self, is_maximized: bool):
        self.maximize_button.setChecked(is_maximized)
        self.maximize_button.setToolTip("Restore" if is_maximized else "Maximize")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
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
            new_window_pos = event.globalPosition().toPoint() - self._mouse_move_offset
            self.window().move(new_window_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._mouse_move_offset = None
        super().mouseReleaseEvent(event)
