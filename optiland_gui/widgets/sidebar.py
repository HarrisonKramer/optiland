# Create optiland_gui/widgets/sidebar.py
# Content:
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QResizeEvent
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QToolButton,
    QSizePolicy,
    QButtonGroup,
)

COLLAPSE_THRESHOLD_WIDTH = 80
SIDEBAR_MIN_WIDTH = 60
SIDEBAR_MAX_WIDTH = 150


class SidebarWidget(QWidget):
    """
    A custom sidebar widget with a title, a series of navigation buttons,
    and a settings button pinned to the bottom.
    The sidebar can collapse to an icon-only mode when its width is reduced.
    """

    menuSelected = Signal(str)  # Emits the objectName of the selected button

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SidebarWidget")
        self.setMinimumWidth(SIDEBAR_MIN_WIDTH)
        self.setMaximumWidth(SIDEBAR_MAX_WIDTH)

        self._is_collapsed = False
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(5, 10, 5, 10) # Adjusted margins for visual appeal
        self._main_layout.setSpacing(5) # Spacing between widgets

        # --- Title Label ---
        self.title_label = QLabel("Optiland")
        self.title_label.setObjectName("SidebarTitleLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setFixedHeight(30) # Adjusted height
        self.title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._main_layout.addWidget(self.title_label)

        self._buttons_list = []  # To store (button, text_label)

        # --- Menu Buttons ---
        button_definitions = [
            ("dash", "Dash", ":/icons/dash.svg"),
            ("design", "Design", ":/icons/design.svg"),
            ("analysis", "Analysis", ":/icons/analysis.svg"),
            ("optimization", "Optimization", ":/icons/optimization.svg"),
            ("materials", "Materials", ":/icons/materials.svg"),
            ("tolerancing", "Tolerancing", ":/icons/tolerancing.svg"),
        ]

        for name, text, icon_path in button_definitions:
            button = QToolButton()
            button.setObjectName(f"sidebar-btn-{name}")
            button.setText(text)
            button.setIcon(QIcon(icon_path))
            button.setIconSize(QSize(24, 24)) 
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon) 
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setFixedHeight(65) 

            self._main_layout.addWidget(button)
            self._button_group.addButton(button)
            self._buttons_list.append({"widget": button, "name": name, "text": text})
            button.clicked.connect(self._handle_button_click)

        self._main_layout.addStretch(1) # Pushes settings button to the bottom

        # --- Settings Button ---
        self.settings_button = QToolButton()
        self.settings_button.setObjectName("sidebar-btn-settings")
        self.settings_button.setText("Settings")
        self.settings_button.setIcon(QIcon(":/icons/settings.svg"))
        self.settings_button.setIconSize(QSize(24, 24))
        self.settings_button.setCheckable(True)
        self.settings_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.settings_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.settings_button.setFixedHeight(35)

        self._main_layout.addWidget(self.settings_button)
        self._button_group.addButton(self.settings_button)
        self._buttons_list.append({"widget": self.settings_button, "name": "settings", "text": "Settings"})
        self.settings_button.clicked.connect(self._handle_button_click)

        # Set "Dash" as active by default
        if self._buttons_list:
            self._buttons_list[0]["widget"].setChecked(True)
            # self.menuSelected.emit(self._buttons_list[0]["name"]) # Optionally emit signal on init

        # Initial check for collapse (e.g. if starting very narrow)
        # This will be more reliably handled by the first resizeEvent or MainWindow's logic
        # self.set_collapsed(self.width() <= COLLAPSE_THRESHOLD_WIDTH)


    def _handle_button_click(self):
        checked_button = self._button_group.checkedButton()
        if checked_button:
            button_name = next((b["name"] for b in self._buttons_list if b["widget"] == checked_button), None)
            if button_name:
                self.menuSelected.emit(button_name)

    def set_collapsed(self, collapsed: bool):
        if self._is_collapsed == collapsed:
            return

        self._is_collapsed = collapsed
        if collapsed:
            self.title_label.setText("O")
            for item in self._buttons_list:
                item["widget"].setText("") # Clear text for icon only mode
                item["widget"].setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
                item["widget"].setToolTip(item["text"])
        else:
            self.title_label.setText("Optiland")
            for item in self._buttons_list:
                item["widget"].setText(item["text"])
                item["widget"].setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
                item["widget"].setToolTip("")
        # Adjust layout/spacing if necessary, but QSS should handle appearance
        self.updateGeometry()


    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        current_width = event.size().width()

        if current_width <= COLLAPSE_THRESHOLD_WIDTH:
            if not self._is_collapsed:
                self.set_collapsed(True)
        else:
            if self._is_collapsed:
                self.set_collapsed(False)

    def force_set_collapse_state(self, collapse: bool):
        """Public method to forcefully set the collapse state, bypassing width check once."""
        self.set_collapsed(collapse)