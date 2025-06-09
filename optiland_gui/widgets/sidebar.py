import webbrowser

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QResizeEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

COLLAPSE_THRESHOLD_WIDTH = 80
SIDEBAR_MIN_WIDTH = 60
SIDEBAR_MAX_WIDTH = 150


class SidebarWidget(QWidget):
    menuSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wip_buttons = [
            "dash",
            "analysis",
            "optimization",
            "materials",
            "tolerancing",
        ]
        self._last_checked_button = None
        self.setObjectName("SidebarWidget")
        self.setMinimumWidth(SIDEBAR_MIN_WIDTH)
        self.setMaximumWidth(SIDEBAR_MAX_WIDTH)

        self._is_collapsed = False
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(5, 10, 5, 10)
        self._main_layout.setSpacing(5)

        self.title_label = QLabel("|||")
        self.title_label.setObjectName("SidebarTitleLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setFixedHeight(30)
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._main_layout.addWidget(self.title_label)

        self._buttons_list = []
        self.current_theme = "dark"

        button_definitions = [
            ("dash", "Dash", "dash.svg"),
            ("design", "Design", "design.svg"),
            ("analysis", "Analysis", "analysis.svg"),
            ("optimization", "Optimization", "optimization.svg"),
            ("materials", "Materials", "materials.svg"),
            ("tolerancing", "Tolerancing", "tolerancing.svg"),
            ("scripts", "Scripts", "terminal.svg"),
        ]

        for name, text, icon_filename in button_definitions:
            button = QToolButton()
            button.setObjectName(f"sidebar-btn-{name}")
            button.setText(text)
            button.setIconSize(QSize(24, 24))
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setFixedHeight(65)

            self._main_layout.addWidget(button)
            self._button_group.addButton(button)
            self._buttons_list.append(
                {
                    "widget": button,
                    "name": name,
                    "text": text,
                    "icon_filename": icon_filename,
                }
            )
            button.clicked.connect(self._handle_button_click)

        self._main_layout.addStretch(1)

        self.settings_button = QToolButton()
        self.settings_button.setObjectName("sidebar-btn-settings")
        self.settings_button.setText("Settings")
        self.settings_button.setIconSize(QSize(24, 24))
        self.settings_button.setCheckable(True)
        self.settings_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextUnderIcon
        )
        self.settings_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.settings_button.setFixedHeight(35)
        self._main_layout.addWidget(self.settings_button)
        self._button_group.addButton(self.settings_button)
        self._buttons_list.append(
            {
                "widget": self.settings_button,
                "name": "settings",
                "text": "Settings",
                "icon_filename": "settings.svg",
            }
        )
        self.settings_button.clicked.connect(self._handle_button_click)

        self.github_button = QToolButton()
        self.github_button.setObjectName("sidebar-btn-github")
        self.github_button.setText("Optiland-GUI")
        self.github_button.setToolTip("Open the Optiland-GUI GitHub page")
        self.github_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextUnderIcon
        )
        self.github_button.clicked.connect(self._open_github_url)
        self.github_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.github_button.setFixedHeight(35)
        self.github_button.setIconSize(QSize(24, 24))
        self._main_layout.addWidget(self.github_button)
        self._buttons_list.append(
            {
                "widget": self.github_button,
                "name": "github",
                "text": "Optiland-GUI",
                "icon_filename": "brand_github.svg",
            }
        )

        self.help_button = QToolButton()
        self.help_button.setObjectName("sidebar-btn-help")
        self.help_button.setText("Help")
        self.help_button.setToolTip("Open the documentation")
        self.help_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.help_button.clicked.connect(self._open_help_url)
        self.help_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.help_button.setFixedHeight(35)
        self.help_button.setIconSize(QSize(24, 24))
        self._main_layout.addWidget(self.help_button)
        self._buttons_list.append(
            {
                "widget": self.help_button,
                "name": "help",
                "text": "Help",
                "icon_filename": "help.svg",
            }
        )

        self.update_icons()

        design_button_item = next(
            (item for item in self._buttons_list if item["name"] == "design"), None
        )
        if design_button_item:
            design_button_item["widget"].setChecked(True)
            self._last_checked_button = design_button_item["widget"]

    def _open_github_url(self):
        url = "https://github.com/HarrisonKramer/optiland"
        webbrowser.open(url)

    def _open_help_url(self):
        url = "https://optiland.readthedocs.io/en/latest/index.html"
        webbrowser.open(url)

    def _handle_button_click(self):
        checked_button = self._button_group.checkedButton()
        if not checked_button:
            return

        button_name = next(
            (b["name"] for b in self._buttons_list if b["widget"] == checked_button),
            None,
        )

        if button_name in self._wip_buttons:
            QMessageBox.information(
                self,
                "Work in Progress",
                "This feature is currently under development.\nStay tuned for "
                "updates to the GUI!",
            )
            checked_button.setChecked(False)
            if self._last_checked_button:
                self._last_checked_button.setChecked(True)
            return

        if button_name:
            self.menuSelected.emit(button_name)
            self._last_checked_button = checked_button

    def set_collapsed(self, collapsed: bool):
        if self._is_collapsed == collapsed:
            return

        self._is_collapsed = collapsed
        if collapsed:
            self.title_label.setText("|||")
            for item in self._buttons_list:
                item["widget"].setText("")
                item["widget"].setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
                item["widget"].setToolTip(item["text"])
        else:
            self.title_label.setText("|||")
            for item in self._buttons_list:
                item["widget"].setText(item["text"])
                item["widget"].setToolButtonStyle(
                    Qt.ToolButtonStyle.ToolButtonTextUnderIcon
                )
                item["widget"].setToolTip("")
        self.updateGeometry()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        current_width = event.size().width()

        if current_width <= COLLAPSE_THRESHOLD_WIDTH and not self._is_collapsed:
            self.set_collapsed(True)
        elif current_width > COLLAPSE_THRESHOLD_WIDTH and self._is_collapsed:
            self.set_collapsed(False)

    def force_set_collapse_state(self, collapse: bool):
        self.set_collapsed(collapse)

    def update_icons(self, theme="dark"):
        self.current_theme = theme
        for item in self._buttons_list:
            icon_path = f":/icons/{self.current_theme}/{item['icon_filename']}"
            item["widget"].setIcon(QIcon(icon_path))
