# Create optiland_gui/widgets/sidebar.py
# Content:
import webbrowser
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QResizeEvent
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QToolButton,
    QSizePolicy,
    QButtonGroup,
    QMessageBox,
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
        self._wip_buttons = ["dash", "analysis", "optimization", "materials", "tolerancing"]
        self._last_checked_button = None
        self.setObjectName("SidebarWidget")
        self.setMinimumWidth(SIDEBAR_MIN_WIDTH)
        self.setMaximumWidth(SIDEBAR_MAX_WIDTH)

        self._is_collapsed = False
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(
            5, 10, 5, 10
        )  # Adjusted margins for visual appeal
        self._main_layout.setSpacing(5)  # Spacing between widgets

        # --- Title Label ---
        self.title_label = QLabel("|||")
        self.title_label.setObjectName("SidebarTitleLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setFixedHeight(30)  # Adjusted height
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._main_layout.addWidget(self.title_label)

        self._buttons_list = []  # To store (button, text_label)
        self.current_theme = "dark"

        # --- Menu Buttons ---
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
            self._buttons_list.append({"widget": button, "name": name, "text": text, "icon_filename": icon_filename})
            button.clicked.connect(self._handle_button_click)

        self._main_layout.addStretch(1)  # Pushes settings button to the bottom

        # --- Settings Button ---
        self.settings_button = QToolButton()
        self.settings_button.setObjectName("sidebar-btn-settings")
        self.settings_button.setText("Settings")
        self.settings_button.setIconSize(QSize(24, 24))
        self.settings_button.setCheckable(True)
        self.settings_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.settings_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.settings_button.setFixedHeight(35)
        self._main_layout.addWidget(self.settings_button)
        self._button_group.addButton(self.settings_button)
        self._buttons_list.append({"widget": self.settings_button, "name": "settings", "text": "Settings", "icon_filename": "settings.svg"})
        self.settings_button.clicked.connect(self._handle_button_click)
        
        # --- GitHub Button ---
        self.github_button = QToolButton()
        self.github_button.setObjectName("sidebar-btn-github")
        self.github_button.setText("Optiland-GUI")
        self.github_button.setToolTip("Open the Optiland-GUI GitHub page")
        self.github_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.github_button.clicked.connect(self._open_github_url)
        self.github_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.github_button.setFixedHeight(35)
        self.github_button.setIconSize(QSize(24, 24))
        self._main_layout.addWidget(self.github_button)
        self._buttons_list.append({"widget": self.github_button, "name": "github", "text": "Optiland-GUI", "icon_filename": "brand_github.svg"})

        # --- Help Button ---
        self.help_button = QToolButton()
        self.help_button.setObjectName("sidebar-btn-help")
        self.help_button.setText("Help")
        self.help_button.setToolTip("Open the documentation")
        self.help_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.help_button.clicked.connect(self._open_help_url)
        self.help_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.help_button.setFixedHeight(35)
        self.help_button.setIconSize(QSize(24, 24))
        self._main_layout.addWidget(self.help_button)
        self._buttons_list.append({"widget": self.help_button, "name": "help", "text": "Help", "icon_filename": "help.svg"})


        self.update_icons()

        # Set "Design" as active by default
        design_button_item = next((item for item in self._buttons_list if item["name"] == "design"), None)
        if design_button_item:
            design_button_item["widget"].setChecked(True)
            self._last_checked_button = design_button_item["widget"]

    def _open_github_url(self):
        """Opens the project's GitHub page in a web browser."""
        url = "https://github.com/HarrisonKramer/optiland" # Replace with your actual URL
        webbrowser.open(url)

    def _open_help_url(self):
        """Opens the project's documentation in a web browser."""
        url = "https://optiland.readthedocs.io/en/latest/index.html" # Replace with your actual URL
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
                "This feature is currently under development.\nStay tuned for updates to the GUI!",
            )
            # Prevent navigation by unchecking the WIP button and re-checking the last valid one
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
                item["widget"].setText("")  # Clear text for icon only mode
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

    def update_icons(self, theme="dark"):
        self.current_theme = theme
        for item in self._buttons_list:
            icon_path = f":/icons/{self.current_theme}/{item['icon_filename']}"
            item['widget'].setIcon(QIcon(icon_path))