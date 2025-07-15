# optiland_gui/widgets/python_terminal.py
import os

from PySide6.QtCore import QRegularExpression, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QSyntaxHighlighter,
    QTextCharFormat,
)
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget


# --- Theme-Aware Python Syntax Highlighter ---
class PythonHighlighter(QSyntaxHighlighter):
    """A theme-aware syntax highlighter for Python code."""

    def __init__(self, parent, theme_colors):
        super().__init__(parent)
        self.theme_colors = theme_colors
        self.highlighting_rules = []
        self._build_rules()

    def _build_rules(self):
        self.highlighting_rules.clear()
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(self.theme_colors["keyword"])
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
            self.highlighting_rules.append((QRegularExpression(word), keyword_format))
        string_format = QTextCharFormat()
        string_format.setForeground(self.theme_colors["string"])
        self.highlighting_rules.append((QRegularExpression('".*"'), string_format))
        self.highlighting_rules.append((QRegularExpression("'.*'"), string_format))
        comment_format = QTextCharFormat()
        comment_format.setForeground(self.theme_colors["comment"])
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((QRegularExpression("#[^\n]*"), comment_format))
        number_format = QTextCharFormat()
        number_format.setForeground(self.theme_colors["number"])
        self.highlighting_rules.append(
            (QRegularExpression("\\b[0-9]+\\.?[0-9]*\\b"), number_format)
        )

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)

    def update_theme(self, theme_colors):
        self.theme_colors = theme_colors
        self._build_rules()
        self.rehighlight()


# --- Code Snippets ---
CODE_SNIPPETS = {
    "Add Standard Surface": "optic = connector.get_optic()\noptic.add_surface(\n    radius=100.0,\n    thickness=5.0,\n    material='N-BK7',\n    comment='New Lens Surface'\n)\n# The GUI will update automatically!",
    "Set System Aperture": "optic = connector.get_optic()\noptic.set_aperture(ap_type='EPD', value=20.0)\nprint(f'Aperture set to: {optic.aperture}')",
    "Get Surface Info": "optic = connector.get_optic()\nsurface = optic.surface_group.surfaces[1]\nprint(f'Surface 1 Radius: {surface.geometry.radius}')",
    "Loop Through Surfaces": "optic = connector.get_optic()\nfor i, surface in enumerate(optic.surface_group.surfaces):\n    print(f'Surface {i}: {surface.comment}, Radius: {surface.geometry.radius}')",
    "Run Analysis": "analysis_panel = iface.get_analysis_panel()\nif analysis_panel:\n    analysis_panel.run_analysis_slot()",
    "Refresh GUI": "# Use this if you make changes outside the connector's methods\niface.refresh_all()",
}


# FIX 2: Custom label to handle left and right clicks
class ClickableLabel(QLabel):
    leftClicked = Signal()
    rightClicked = Signal(object)  # Will emit the event position

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.leftClicked.emit()
        elif event.button() == Qt.RightButton:
            self.rightClicked.emit(event.pos())
        super().mousePressEvent(event)


class PythonTerminalWidget(QWidget):
    commandExecuted = Signal()
    ICON_SIZE = QSize(22, 22)

    def __init__(self, parent=None, custom_variables=None, theme="dark"):
        super().__init__(parent)
        self.setObjectName("PythonTerminalWidget")
        self.current_theme = theme
        self.untitled_script_counter = 0

        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        self.injected_variables = custom_variables if custom_variables else {}
        if self.injected_variables:
            self.kernel_manager.kernel.shell.push(self.injected_variables)

        self.kernel_manager.kernel.shell.events.register(
            "post_execute", self._on_kernel_execute
        )

        self._define_themes()
        self._setup_ui()
        self.set_theme(self.current_theme)

    def _define_themes(self):
        self.themes = {
            "dark": {
                "keyword": QColor("#CF8A2E"),
                "string": QColor("#A2E05D"),
                "comment": QColor("#9E9E9E"),
                "number": QColor("#B5CEA8"),
                "editor_bg": "#2B2B2B",
                "editor_fg": "#F8F8F2",
                "jupyter_style": "monokai",
                "btn_hover": "#555555",
                "splitter": "#555555",
                "title_snippets": "#F8F8F2",
            },
            "light": {
                "keyword": QColor("#0000FF"),
                "string": QColor("#008000"),
                "comment": QColor("#808080"),
                "number": QColor("#A31515"),
                "editor_bg": "#FFFFFF",
                "editor_fg": "#000000",
                "jupyter_style": "default",
                "btn_hover": "#EEEEEE",
                "splitter": "#FFFFFF",
                "title_snippets": "#2B2B2B",
            },
        }

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.dock_area = QMainWindow()
        self.dock_area.setWindowFlags(Qt.Widget)
        self.dock_area.setDockNestingEnabled(True)
        main_layout.addWidget(self.dock_area)
        self.console_dock = self._create_console_dock()
        self.dock_area.addDockWidget(Qt.RightDockWidgetArea, self.console_dock)
        self.editor_dock = self._create_editor_dock()
        self.dock_area.addDockWidget(Qt.LeftDockWidgetArea, self.editor_dock)

    def _create_editor_dock(self):
        dock = QDockWidget("Script Editor", self)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        editor_widget = QWidget()
        layout = QVBoxLayout(editor_widget)
        layout.setContentsMargins(5, 5, 5, 5)

        toolbar_layout = QHBoxLayout()
        self.btn_new_script = QPushButton(toolTip="New Script (Ctrl+N)")
        self.btn_run_script = QPushButton(toolTip="Run Script (F5)")
        self.btn_save_script = QPushButton(toolTip="Save Script (Ctrl+S)")
        self.btn_load_script = QPushButton(toolTip="Load Script (Ctrl+O)")
        self.btn_save_quick_action = QPushButton(toolTip="Save as Quick-Action")
        for btn in [
            self.btn_new_script,
            self.btn_run_script,
            self.btn_save_script,
            self.btn_load_script,
            self.btn_save_quick_action,
        ]:
            btn.setIconSize(self.ICON_SIZE)
            btn.setFlat(True)
        toolbar_layout.addWidget(self.btn_new_script)
        toolbar_layout.addWidget(self.btn_run_script)
        toolbar_layout.addWidget(self.btn_save_script)
        toolbar_layout.addWidget(self.btn_load_script)
        toolbar_layout.addWidget(self.btn_save_quick_action)
        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)

        editor_area_layout = QHBoxLayout()
        layout.addLayout(editor_area_layout)

        self.script_tabs = QTabWidget()
        self.script_tabs.setTabsClosable(True)
        self.script_tabs.setMovable(True)
        self.script_tabs.setObjectName("ScriptTabWidget")
        editor_area_layout.addWidget(self.script_tabs, 1)

        self.snippets_panel = QWidget()
        self.snippets_panel.setFixedWidth(200)
        snippets_layout = QVBoxLayout(self.snippets_panel)
        snippets_layout.setContentsMargins(4, 0, 0, 0)

        snippets_layout.setAlignment(Qt.AlignTop)

        self.snippets_title_label = ClickableLabel("Quick-Actions")
        snippets_layout.addWidget(self.snippets_title_label)

        self.snippets_list = QListWidget()
        snippets_layout.addWidget(self.snippets_list)

        self._refresh_snippets_list()
        editor_area_layout.addWidget(self.snippets_panel)

        self._create_new_tab()
        dock.setWidget(editor_widget)

        self.btn_new_script.clicked.connect(lambda: self._create_new_tab())
        self.btn_run_script.clicked.connect(self._run_script_from_editor)
        self.btn_save_script.clicked.connect(lambda: self._save_script())
        self.btn_load_script.clicked.connect(self._load_script)
        # FIX 3: Correctly connect the button to its function
        self.btn_save_quick_action.clicked.connect(self._save_as_quick_action)
        self.snippets_list.itemDoubleClicked.connect(self._insert_snippet_from_item)
        self.script_tabs.tabCloseRequested.connect(self._close_tab)

        # Connect the custom signals from the clickable label
        self.snippets_title_label.leftClicked.connect(self._toggle_snippets_collapse)
        self.snippets_title_label.rightClicked.connect(self._show_icon_context_menu)

        self.snippets_list.setVisible(True)

        return dock

    def _create_console_dock(self):
        dock = QDockWidget("Console", self)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.jupyter_widget = RichJupyterWidget()
        self.jupyter_widget.kernel_manager = self.kernel_manager
        self.jupyter_widget.kernel_client = self.kernel_client
        self.jupyter_widget.banner = (
            "Welcome to Optiland!\n'connector' and 'iface' are available.\n"
        )
        dock.setWidget(self.jupyter_widget)
        return dock

    def _get_current_editor(self):
        return self.script_tabs.currentWidget()

    def _create_new_tab(self, content="", file_path=None):
        self.untitled_script_counter += 1

        editor = QTextEdit()
        editor.setFont(QFont("Cascadia Code", 10))
        editor.setLineWrapMode(QTextEdit.NoWrap)
        editor.setPlainText(content)

        theme_colors = self.themes.get(self.current_theme, self.themes["dark"])
        highlighter = PythonHighlighter(editor.document(), theme_colors)
        editor.setProperty("highlighter", highlighter)

        # Track modification state and file path for each tab
        editor.setProperty("is_modified", bool(content) and file_path is None)
        editor.setProperty("file_path", file_path)
        editor.textChanged.connect(lambda: self._on_text_changed(editor))

        # Set tab title
        title = (
            os.path.basename(file_path)
            if file_path
            else f"Untitled-{self.untitled_script_counter}"
        )
        index = self.script_tabs.addTab(editor, title)
        self.script_tabs.setTabText(index, title)

        self.script_tabs.setCurrentIndex(index)
        self.set_theme(self.current_theme)
        return editor

    def _on_text_changed(self, editor):
        """Mark the tab as modified and add an asterisk to the title."""
        if editor and not editor.property("is_modified"):
            editor.setProperty("is_modified", True)
            current_index = self.script_tabs.indexOf(editor)
            current_title = self.script_tabs.tabText(current_index)
            if not current_title.endswith("*"):
                self.script_tabs.setTabText(current_index, current_title + "*")

    def _close_tab(self, index):
        editor = self.script_tabs.widget(index)
        if not editor or not editor.property("is_modified"):
            self._perform_close_tab(index)
            return

        tab_title = self.script_tabs.tabText(index).replace("*", "")
        reply = QMessageBox.warning(
            self,
            "Save Changes",
            f"The script '{tab_title}' has been modified.\n\nDo you want to save your changes?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )

        if reply == QMessageBox.Save:
            if self._save_script(index):
                self._perform_close_tab(index)
        elif reply == QMessageBox.Discard:
            self._perform_close_tab(index)

    def _perform_close_tab(self, index):
        if self.script_tabs.count() > 1:
            self.script_tabs.removeTab(index)
        else:
            self._create_new_tab()
            self.script_tabs.removeTab(index)

    def _refresh_snippets_list(self):
        self.snippets_list.clear()
        for name in CODE_SNIPPETS:
            self.snippets_list.addItem(QListWidgetItem(name))

    def _toggle_snippets_collapse(self):
        icon_name = f":/icons/{self.current_theme}/flash.svg"  # Ensure you have a 'flash_actions.svg' icon
        if self.snippets_list.isVisible():
            self.snippets_list.setVisible(False)
            self.snippets_panel.setFixedWidth(40)
            self.snippets_title_label.setText("")
            self.snippets_title_label.setPixmap(QIcon(icon_name).pixmap(24))
        else:
            self.snippets_list.setVisible(True)
            self.snippets_panel.setFixedWidth(200)
            self.snippets_title_label.setText("Quick-Actions")

    # FIX 2: Context menu for the icon when panel is collapsed
    def _show_icon_context_menu(self, position):
        if not self.snippets_list.isVisible():
            self._show_snippets_context_menu(position)

    def _show_snippets_context_menu(self, position):
        menu = QMenu(self)
        for name in CODE_SNIPPETS:
            action = QAction(name, self)
            action.triggered.connect(
                lambda checked=False, n=name: self._insert_snippet(n)
            )
            menu.addAction(action)

        # Use the global position of the label to show the menu correctly
        global_pos = self.snippets_title_label.mapToGlobal(position)
        menu.exec(global_pos)

    def _on_kernel_execute(self):
        self.commandExecuted.emit()

    def _run_script_from_editor(self):
        editor = self._get_current_editor()
        if editor and editor.toPlainText().strip():
            self.kernel_client.execute(editor.toPlainText(), silent=False)
            self.console_dock.raise_()

    def _insert_snippet_from_item(self, item):
        self._insert_snippet(item.text())

    def _insert_snippet(self, name):
        editor = self._get_current_editor() or self._create_new_tab()
        snippet_code = CODE_SNIPPETS.get(name, "")
        editor.setPlainText(snippet_code)
        editor.setFocus()

    def _save_script(self, index_to_save=None):
        """Saves the script. Can be called for a specific tab index."""
        if index_to_save is None:
            index_to_save = self.script_tabs.currentIndex()

        editor = self.script_tabs.widget(index_to_save)
        if not editor:
            return False

        file_path = editor.property("file_path")
        if not file_path:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Script", "", "Python Files (*.py)"
            )
            if not file_path:
                return False  # User cancelled

        try:
            with open(file_path, "w") as f:
                f.write(editor.toPlainText())

            # Update tab state to be "clean"
            editor.setProperty("is_modified", False)
            editor.setProperty("file_path", file_path)
            title = os.path.basename(file_path)
            self.script_tabs.setTabText(index_to_save, title)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save file:\n{e}")
            return False

    def _load_script(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Script", "", "Python Files (*.py)"
        )
        if not filepath:
            return

        with open(filepath) as f:
            content = f.read()

        editor = self._get_current_editor()
        if (
            editor
            and not editor.toPlainText().strip()
            and not editor.property("is_modified")
        ):
            editor.setPlainText(content)
            editor.setProperty("file_path", filepath)
            editor.setProperty("is_modified", False)
            title = os.path.basename(filepath)
            self.script_tabs.setTabText(self.script_tabs.currentIndex(), title)
        else:
            self._create_new_tab(content=content, file_path=filepath)

    def _save_as_quick_action(self):
        editor = self._get_current_editor()
        if not editor or not editor.toPlainText().strip():
            return
        text, ok = QInputDialog.getText(
            self, "Save Quick-Action", "Enter a name for this action:"
        )
        if ok and text:
            CODE_SNIPPETS[text] = editor.toPlainText()
            self._refresh_snippets_list()

    def set_theme(self, theme_name="dark"):
        self.current_theme = theme_name
        theme_palette = self.themes.get(theme_name, self.themes["dark"])

        btn_style = f"QPushButton {{ background-color: transparent; border: none; padding: 2px; }} QPushButton:hover {{ background-color: {theme_palette['btn_hover']}; border-radius: 3px; }}"
        for btn in [
            self.btn_new_script,
            self.btn_run_script,
            self.btn_save_script,
            self.btn_load_script,
            self.btn_save_quick_action,
        ]:
            btn.setStyleSheet(btn_style)

        # FIX 1: Apply theme color to the Quick-Actions title
        self.snippets_title_label.setStyleSheet(
            f"font-weight: bold; padding: 2px; color: {theme_palette['title_snippets']};"
        )

        # Make sure icon names match your resource file
        self.btn_new_script.setIcon(QIcon(f":/icons/{theme_name}/add.svg"))
        self.btn_run_script.setIcon(QIcon(f":/icons/{theme_name}/run.svg"))
        self.btn_save_script.setIcon(QIcon(f":/icons/{theme_name}/save_settings.svg"))
        self.btn_load_script.setIcon(QIcon(f":/icons/{theme_name}/load_settings.svg"))
        self.btn_save_quick_action.setIcon(
            QIcon(f":/icons/{theme_name}/add_quick_action.svg")
        )
        self.script_tabs.setStyleSheet(
            f"QTabBar::close-button {{ image: url(:/icons/{theme_name}/close.svg); }} QTabBar::close-button:hover {{ background: #DB0000; }}"
        )

        self.jupyter_widget.syntax_style = theme_palette["jupyter_style"]
        self.jupyter_widget._style_sheet_changed()
        self.jupyter_widget.setStyleSheet(
            f"QWidget {{ background-color: {theme_palette['editor_bg']}; color: {theme_palette['editor_fg']}; }}"
        )

        editor_style = f"QTextEdit {{ background-color: {theme_palette['editor_bg']}; color: {theme_palette['editor_fg']}; font-family: Cascadia Code; }}"
        for i in range(self.script_tabs.count()):
            editor = self.script_tabs.widget(i)
            if not editor:
                continue
            editor.setStyleSheet(editor_style)
            highlighter = editor.property("highlighter")
            if highlighter:
                highlighter.update_theme(theme_palette)

        if not self.snippets_list.isVisible():
            icon_name = f":/icons/{self.current_theme}/flash.svg"
            self.snippets_title_label.setPixmap(QIcon(icon_name).pixmap(24))

    def shutdown_kernel(self):
        if self.kernel_client and self.kernel_client.channels_running:
            self.kernel_client.stop_channels()
        if self.kernel_manager and self.kernel_manager.is_alive():
            self.kernel_manager.shutdown_kernel()
