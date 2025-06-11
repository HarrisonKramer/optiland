# optiland_gui/widgets/python_terminal.py
import sys
from io import StringIO

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# QScintilla is a third-party dependency.
# Please install it using: pip install PyQt6-QScintilla
try:
    from Qsci import QsciLexerPython, QsciScintilla
    QSCINTILLA_AVAILABLE = True
except ImportError:
    QSCINTILLA_AVAILABLE = False

from qtconsole.inprocess import QtInProcessKernelManager

# --- Code Snippets ---
# A dictionary of helpful code snippets for users.
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

        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()

        self.kernel = self.kernel_manager.kernel
        self.kernel.shell.events.register("post_execute", self.on_command_executed)

        # Inject variables into the kernel's namespace
        self.injected_variables = custom_variables if custom_variables else {}
        if self.injected_variables:
            self.kernel.shell.push(self.injected_variables)

        self._setup_ui()
        self.set_theme(theme)
        self._print_welcome_message()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        # --- Toolbar ---
        toolbar_layout = QHBoxLayout()
        self.btn_run_script = QPushButton("Run Script")
        self.btn_save_script = QPushButton("Save Script")
        self.btn_load_script = QPushButton("Load Script")
        toolbar_layout.addWidget(self.btn_run_script)
        toolbar_layout.addWidget(self.btn_save_script)
        toolbar_layout.addWidget(self.btn_load_script)
        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)

        # --- Main Splitter ---
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(main_splitter)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # --- Code Editor ---
        if QSCINTILLA_AVAILABLE:
            self.editor = QsciScintilla()
            self.editor.setUtf8(True)
            self.editor.setFont(self.font())
            lexer = QsciLexerPython()
            self.editor.setLexer(lexer)
            # Simple auto-completion based on all words in the document
            self.editor.setAutoCompletionSource(QsciScintilla.AcsDocument)
            self.editor.setAutoCompletionThreshold(1)
            # Brace matching
            self.editor.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
            # Indentation
            self.editor.setIndentationsUseTabs(False)
            self.editor.setTabWidth(4)
            self.editor.setIndentationGuides(True)
            self.editor.setAutoIndent(True)
        else:
            self.editor = QTextEdit()
            self.editor.setPlaceholderText(
                "QScintilla not found. Please install PyQt6-QScintilla for a better experience."
            )
        top_layout.addWidget(self.editor, 4)  # Give editor more space initially

        # --- Snippets ---
        self.snippets_list = QListWidget()
        self.snippets_list.setMaximumWidth(200)
        for name in CODE_SNIPPETS:
            self.snippets_list.addItem(QListWidgetItem(name))
        top_layout.addWidget(self.snippets_list, 1)

        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # --- Console Output ---
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        bottom_layout.addWidget(self.console_output)

        # --- Console Input ---
        self.console_input = QLineEdit()
        self.console_input.setPlaceholderText("Type a command and press Enter...")
        bottom_layout.addWidget(self.console_input)

        main_splitter.addWidget(top_widget)
        main_splitter.addWidget(bottom_widget)
        main_splitter.setSizes([300, 200]) # Initial sizing

        # --- Connections ---
        self.btn_run_script.clicked.connect(self.run_script)
        self.btn_save_script.clicked.connect(self.save_script)
        self.btn_load_script.clicked.connect(self.load_script)
        self.console_input.returnPressed.connect(self.run_console_command)
        self.snippets_list.itemDoubleClicked.connect(self.insert_snippet)

    def _print_welcome_message(self):
        welcome = (
            "Welcome to the Optiland Python Console!\n\n"
            "The following objects are available in the global namespace:\n"
            "  - 'connector': The main bridge to the optical system.\n"
            "     Usage: `connector.add_surface(...)`\n"
            "     Try: `help(connector)`\n\n"
            "  - 'iface': The interface to control the GUI itself.\n"
            "     Usage: `iface.show_lens_editor()`\n"
            "     Try: `help(iface)`\n\n"
            "Double-click a snippet on the right to load it into the editor.\n"
            "-----------------------------------------------------------\n"
        )
        self.console_output.setText(welcome)

    def run_script(self):
        """Executes the entire content of the editor pane."""
        script_text = self.editor.text() if QSCINTILLA_AVAILABLE else self.editor.toPlainText()
        if script_text:
            self.execute_code(script_text)

    def run_console_command(self):
        """Executes the command from the console input line."""
        command = self.console_input.text()
        if command:
            self.console_input.clear()
            self.execute_code(command, is_single_line=True)

    def insert_snippet(self, item):
        """Inserts a predefined code snippet into the editor."""
        snippet_name = item.text()
        snippet_code = CODE_SNIPPETS.get(snippet_name, "")
        if QSCINTILLA_AVAILABLE:
            self.editor.setText(snippet_code)
        else:
            self.editor.setPlainText(snippet_code)

    def execute_code(self, code, is_single_line=False):
        """Execute code in the kernel and display the output."""
        if is_single_line:
            self.console_output.append(f">>> {code}")
        else:
            self.console_output.append(">>> Running Script...")

        # Redirect stdout and stderr to capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = captured_stdout = StringIO()
        sys.stderr = captured_stderr = StringIO()

        try:
            # Execute the code in the kernel's namespace
            self.kernel.shell.run_cell(code, store_history=True)

            # Get output
            stdout_val = captured_stdout.getvalue()
            stderr_val = captured_stderr.getvalue()

            if stdout_val:
                self.console_output.append(stdout_val.strip())
            if stderr_val:
                self.console_output.setTextColor(Qt.GlobalColor.red)
                self.console_output.append(stderr_val.strip())
                self.console_output.setTextColor(self.default_text_color)

        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        self.console_output.ensureCursorVisible()

    def on_command_executed(self):
        """Fires after any command is executed by the kernel."""
        self.commandExecuted.emit()

    def set_theme(self, theme="dark"):
        """Applies a color theme to the terminal widgets."""
        if theme == "dark":
            self.default_text_color = Qt.GlobalColor.lightGray
            bg_color = "#2B2B2B"
            text_color = "#F8F8F2"
            selection_bg = "#44475A"
        else:
            self.default_text_color = Qt.GlobalColor.black
            bg_color = "#FFFFFF"
            text_color = "#000000"
            selection_bg = "#AAD5FF"

        style = (
            f"background-color: {bg_color}; "
            f"color: {text_color}; "
            f"selection-background-color: {selection_bg};"
        )
        self.console_output.setStyleSheet(style)
        self.console_input.setStyleSheet(style)
        if QSCINTILLA_AVAILABLE:
            # Themes for QScintilla are more complex, this is a basic setup
            self.editor.setPaper(Qt.GlobalColor.black if theme == "dark" else Qt.GlobalColor.white)
            self.editor.lexer().setColor(Qt.GlobalColor.white if theme == "dark" else Qt.GlobalColor.black)
            self.editor.lexer().setPaper(Qt.GlobalColor.black if theme == "dark" else Qt.GlobalColor.white)
            self.editor.setCaretForegroundColor(Qt.GlobalColor.white if theme == "dark" else Qt.GlobalColor.black)


    def save_script(self):
        """Saves the editor content to a Python file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Script", "", "Python Files (*.py);;All Files (*)"
        )
        if filepath:
            script_text = self.editor.text() if QSCINTILLA_AVAILABLE else self.editor.toPlainText()
            with open(filepath, 'w') as f:
                f.write(script_text)

    def load_script(self):
        """Loads a Python file into the editor."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Script", "", "Python Files (*.py);;All Files (*)"
        )
        if filepath:
            with open(filepath, 'r') as f:
                script_text = f.read()
            if QSCINTILLA_AVAILABLE:
                self.editor.setText(script_text)
            else:
                self.editor.setPlainText(script_text)

    def shutdown_kernel(self):
        """Shuts down the kernel manager cleanly."""
        if self.kernel_manager and self.kernel_manager.is_alive():
            self.kernel_manager.shutdown_kernel()