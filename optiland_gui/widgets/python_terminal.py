import os

from PySide6.QtCore import QRegularExpression, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget


# --- Simple Python Syntax Highlighter for QTextEdit ---
class PythonHighlighter(QSyntaxHighlighter):
    """
    A simple syntax highlighter for Python code in a QTextEdit widget.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.highlighting_rules = []

        # Keyword format
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#CF8A2E"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "\\bFalse\\b",
            "\\bNone\\b",
            "\\bTrue\\b",
            "\\band\\b",
            "\\bas\\b",
            "\\bassert\\b",
            "\\bbreak\\b",
            "\\bclass\\b",
            "\\bcontinue\\b",
            "\\bdef\\b",
            "\\bdel\\b",
            "\\belif\\b",
            "\\belse\\b",
            "\\bexcept\\b",
            "\\bfinally\\b",
            "\\bfor\\b",
            "\\bfrom\\b",
            "\\bglobal\\b",
            "\\bif\\b",
            "\\bimport\\b",
            "\\bin\\b",
            "\\bis\\b",
            "\\blambda\\b",
            "\\bnonlocal\\b",
            "\\bnot\\b",
            "\\bor\\b",
            "\\bpass\\b",
            "\\braise\\b",
            "\\breturn\\b",
            "\\btry\\b",
            "\\bwhile\\b",
            "\\bwith\\b",
            "\\byield\\b",
            "\\bself\\b",
        ]
        for word in keywords:
            rule = (QRegularExpression(word), keyword_format)
            self.highlighting_rules.append(rule)

        # String format (single and double quoted)
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#A2E05D"))  # Green for strings
        self.highlighting_rules.append((QRegularExpression('".*"'), string_format))
        self.highlighting_rules.append((QRegularExpression("'.*'"), string_format))

        # Comment format
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#9E9E9E"))  # Gray for comments
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((QRegularExpression("#[^\n]*"), comment_format))

        # Number format
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#B5CEA8"))  # Light green/blue for numbers
        self.highlighting_rules.append(
            (QRegularExpression("\\b[0-9]+\\.?[0-9]*\\b"), number_format)
        )

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)


# --- Code Snippets ---
CODE_SNIPPETS = {
    "Add Standard Surface": "optic = connector.get_optic()\n"
    "optic.add_surface(\n"
    "    radius=100.0,\n"
    "    thickness=5.0,\n"
    "    material='N-BK7',\n"
    "    comment='New Lens Surface'\n"
    ")\n"
    "# The GUI will update automatically!",
    "Set System Aperture": "optic = connector.get_optic()\n"
    "optic.set_aperture(ap_type='EPD', value=20.0)\n"
    "print(f'Aperture set to: {optic.aperture}')",
    "Get Surface Info": "optic = connector.get_optic()\n"
    "surface = optic.surface_group.surfaces[1]\n"
    "print(f'Surface 1 Radius: {surface.geometry.radius}')",
    "Loop Through Surfaces": "optic = connector.get_optic()\n"
    "for i, surface in enumerate(optic.surface_group.surfaces):\n"
    "    print(f'Surface {i}: {surface.comment}, Radius: {surface.geometry.radius}')",
    "Run Analysis": "analysis_panel = iface.get_analysis_panel()\n"
    "if analysis_panel:\n"
    "    analysis_panel.run_analysis_slot()",
    "Refresh GUI": "# Use this if you make changes outside the connector's methods\n"
    "iface.refresh_all()",
}


class PythonTerminalWidget(QWidget):
    commandExecuted = Signal()

    def __init__(self, parent=None, custom_variables=None, theme="dark"):
        super().__init__(parent)
        self.setObjectName("PythonTerminalWidget")
        self.current_theme = theme

        # --- Kernel Setup
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        # Inject variables into the kernel
        self.injected_variables = custom_variables if custom_variables else {}
        if self.injected_variables:
            self.kernel_manager.kernel.shell.push(self.injected_variables)

        self.kernel_manager.kernel.shell.events.register(
            "post_execute", self._on_kernel_execute
        )

        self._setup_ui()
        self.set_theme(self.current_theme)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.dock_area = QMainWindow()
        self.dock_area.setWindowFlags(Qt.Widget)
        self.dock_area.setDockNestingEnabled(True)
        main_layout.addWidget(self.dock_area)

        # --- Create Editor Dock
        self.editor_dock = self._create_editor_dock()
        self.dock_area.addDockWidget(Qt.LeftDockWidgetArea, self.editor_dock)

        # --- Create Console Dock
        self.console_dock = self._create_console_dock()
        self.dock_area.addDockWidget(Qt.RightDockWidgetArea, self.console_dock)

    def _create_editor_dock(self):
        """Creates the dock widget for the 'Script Editor'."""
        dock = QDockWidget("Script Editor", self)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)

        editor_widget = QWidget()
        layout = QVBoxLayout(editor_widget)
        layout.setContentsMargins(5, 5, 5, 5)

        # script actions toolbar
        toolbar_layout = QHBoxLayout()
        self.btn_run_script = QPushButton()
        self.btn_run_script.setToolTip("Run Script (F5)")
        self.btn_save_script = QPushButton()
        self.btn_save_script.setToolTip("Save Script (Ctrl+S)")
        self.btn_load_script = QPushButton()
        self.btn_load_script.setToolTip("Load Script (Ctrl+O)")

        for btn in [self.btn_run_script, self.btn_save_script, self.btn_load_script]:
            btn.setIconSize(QSize(24, 24))
            btn.setFlat(True)
            btn.setStyleSheet("border: none; padding: 2px;")

        toolbar_layout.addWidget(self.btn_run_script)
        toolbar_layout.addWidget(self.btn_save_script)
        toolbar_layout.addWidget(self.btn_load_script)
        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # The Code Editor is a simple QTextEdit
        self.editor = QTextEdit()
        self.editor.setFont(QFont("Courier", 10))
        self.editor.setLineWrapMode(QTextEdit.NoWrap)
        self.highlighter = PythonHighlighter(self.editor.document())
        splitter.addWidget(self.editor)

        # --- Quick Actions panel
        snippets_panel = QWidget()
        snippets_layout = QVBoxLayout(snippets_panel)
        snippets_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("Quick-Actions")
        title_label.setStyleSheet("font-weight: bold; padding: 2px;")
        snippets_layout.addWidget(title_label)

        self.snippets_list = QListWidget()
        snippets_layout.addWidget(self.snippets_list)
        for name in CODE_SNIPPETS:
            self.snippets_list.addItem(QListWidgetItem(name))

        splitter.addWidget(snippets_panel)

        splitter.setSizes([600, 200])

        # Set the main widget
        dock.setWidget(editor_widget)

        # --- Connections
        self.btn_run_script.clicked.connect(self._run_script_from_editor)
        self.btn_save_script.clicked.connect(self._save_script)
        self.btn_load_script.clicked.connect(self._load_script)
        self.snippets_list.itemDoubleClicked.connect(self._insert_snippet)

        return dock

    def _create_console_dock(self):
        """Creates the dock widget for the interactive console."""
        dock = QDockWidget("Console", self)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)

        self.jupyter_widget = RichJupyterWidget()
        self.jupyter_widget.kernel_manager = self.kernel_manager
        self.jupyter_widget.kernel_client = self.kernel_client

        # Set a welcome banner - to be changed later
        banner = (
            "Welcome to the Optiland Python Console!\n\n"
            "This is an interactive IPython console.\n"
            "The following objects are available:\n"
            "  - 'connector': The main bridge to the optical system.\n"
            "     Try: `help(connector)`\n\n"
            "  - 'iface': The interface to control the GUI itself.\n"
            "     Try: `help(iface)`\n\n"
            "Output from the 'Script Editor' will also appear here.\n"
            "-----------------------------------------------------------\n"
        )
        self.jupyter_widget.banner = banner
        dock.setWidget(self.jupyter_widget)
        return dock

    def _on_kernel_execute(self):
        """Fires after any command is executed by the kernel."""
        self.commandExecuted.emit()

    def _run_script_from_editor(self):
        """Executes the script from the editor in the shared kernel."""
        script_text = self.editor.toPlainText()

        if script_text.strip():
            self.kernel_client.execute(script_text, silent=False)
            self.console_dock.raise_()  # Bring console to front to show output

    def _insert_snippet(self, item):
        """Inserts a predefined code snippet into the editor."""
        snippet_name = item.text()
        snippet_code = CODE_SNIPPETS.get(snippet_name, "")
        self.editor.setPlainText(snippet_code)
        self.editor.setFocus()

    def _save_script(self):
        """Saves the editor content to a Python file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Script", "", "Python Files (*.py);;All Files (*)"
        )
        if filepath:
            with open(filepath, "w") as f:
                f.write(self.editor.toPlainText())

    def _load_script(self):
        """Loads a Python file into the editor."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Script", "", "Python Files (*.py);;All Files (*)"
        )
        if filepath:
            with open(filepath) as f:
                self.editor.setPlainText(f.read())

    def _update_button_icons(self, theme_name):
        """Updates the icons for the toolbar buttons based on the theme."""
        base_path = os.path.join(
            os.path.dirname(__file__), "resources", "icons", theme_name
        )

        self.btn_run_script.setIcon(QIcon(os.path.join(base_path, "script_run.svg")))
        self.btn_save_script.setIcon(QIcon(os.path.join(base_path, "script_save.svg")))
        self.btn_load_script.setIcon(QIcon(os.path.join(base_path, "script_load.svg")))

    def set_theme(self, theme="dark"):
        """Applies a color theme to the terminal widgets."""
        self.current_theme = theme
        self._update_button_icons(theme)

        if theme == "dark":
            console_bg_color = "#272822"

            editor_bg_color = "#2B2B2B"
            editor_fg_color = "#F8F8F2"

            self.jupyter_widget.syntax_style = "monokai"
            self.jupyter_widget.style_sheet = f"background-color: {console_bg_color};"
            self.editor.setStyleSheet(
                f"background-color: {editor_bg_color}; "
                f"color: {editor_fg_color}; font-family: Courier;"
            )
        else:
            console_bg_color = "#FFFFFF"

            editor_bg_color = "#FDFDFD"
            editor_fg_color = "#000000"

            self.jupyter_widget.syntax_style = "default"
            self.jupyter_widget.style_sheet = f"background-color: {console_bg_color};"
            self.editor.setStyleSheet(
                f"background-color: {editor_bg_color}; "
                f"color: {editor_fg_color}; font-family: Courier;"
            )

        self.highlighter.rehighlight()

    def shutdown_kernel(self):
        """Shuts down the kernel manager cleanly."""
        if self.kernel_client:
            self.kernel_client.stop_channels()
        if self.kernel_manager:
            self.kernel_manager.shutdown_kernel()
