# optiland_gui/main_window.py
import os

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
    Slot,
)  # Added QRect
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
)

from .analysis_panel import AnalysisPanel
from .lens_editor import LensEditor
from .optiland_connector import OptilandConnector
from .optimization_panel import OptimizationPanel
from .system_properties_panel import SystemPropertiesPanel
from .viewer_panel import ViewerPanel

THEME_DARK_PATH = os.path.join(os.path.dirname(__file__), "resources", "dark_theme.qss")
THEME_LIGHT_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "light_theme.qss"
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optiland GUI")
        self.setGeometry(100, 100, 1600, 900)

        self.connector = OptilandConnector()
        self.dock_animations = {}  # To store animations
        self.dock_original_sizes = {}  # To store original sizes for restore

        # --- Create Panels ---
        self.lensEditor = LensEditor(self.connector)
        self.viewerPanel = ViewerPanel(self.connector)
        self.analysisPanel = AnalysisPanel(self.connector)
        self.optimizationPanel = OptimizationPanel(self.connector)
        self.systemPropertiesPanel = SystemPropertiesPanel(self.connector)

        self.setCentralWidget(self.viewerPanel)

        # --- Dock Widgets ---
        self.lensEditorDock = QDockWidget("Lens Data Editor", self)
        self.lensEditorDock.setWidget(self.lensEditor)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.lensEditorDock)
        # Store initial desired width (can be refined)
        self.dock_original_sizes[self.lensEditorDock] = (
            self.lensEditorDock.sizeHint().width()
            if self.lensEditorDock.sizeHint().width() > 0
            else 300
        )

        self.systemPropertiesDock = QDockWidget("System Properties", self)
        self.systemPropertiesDock.setWidget(self.systemPropertiesPanel)
        self.tabifyDockWidget(self.lensEditorDock, self.systemPropertiesDock)
        self.lensEditorDock.raise_()
        self.dock_original_sizes[self.systemPropertiesDock] = (
            self.systemPropertiesDock.sizeHint().width()
            if self.systemPropertiesDock.sizeHint().width() > 0
            else 300
        )

        self.analysisPanelDock = QDockWidget("Analysis", self)
        self.analysisPanelDock.setWidget(self.analysisPanel)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.analysisPanelDock
        )
        self.dock_original_sizes[self.analysisPanelDock] = (
            self.analysisPanelDock.sizeHint().width()
            if self.analysisPanelDock.sizeHint().width() > 0
            else 300
        )

        self.optimizationPanelDock = QDockWidget("Optimization", self)
        self.optimizationPanelDock.setWidget(self.optimizationPanel)
        self.tabifyDockWidget(self.analysisPanelDock, self.optimizationPanelDock)
        self.analysisPanelDock.raise_()
        self.dock_original_sizes[self.optimizationPanelDock] = (
            self.optimizationPanelDock.sizeHint().width()
            if self.optimizationPanelDock.sizeHint().width() > 0
            else 300
        )

        self._create_actions()
        self._create_menu_bar()
        self._create_tool_bars()

        self.current_theme_path = THEME_DARK_PATH
        self.load_stylesheet(self.current_theme_path)

        # Connect toggle actions for animation
        self.lensEditorDock.toggleViewAction().triggered.disconnect()  # Disconnect default
        self.lensEditorDock.toggleViewAction().triggered.connect(
            lambda checked, dock=self.lensEditorDock: self.animate_dock_toggle(
                dock, checked
            )
        )
        self.systemPropertiesDock.toggleViewAction().triggered.disconnect()
        self.systemPropertiesDock.toggleViewAction().triggered.connect(
            lambda checked, dock=self.systemPropertiesDock: self.animate_dock_toggle(
                dock, checked
            )
        )
        # For tabified docks, the logic might need to be smarter,
        # as only one toggle action (for the currently visible tab's dock) might be relevant,
        # or we animate the parent QTabBar/QDockWidget if that's the visual container.
        # For now, let's connect them all and see.
        self.analysisPanelDock.toggleViewAction().triggered.disconnect()
        self.analysisPanelDock.toggleViewAction().triggered.connect(
            lambda checked, dock=self.analysisPanelDock: self.animate_dock_toggle(
                dock, checked
            )
        )
        self.optimizationPanelDock.toggleViewAction().triggered.disconnect()
        self.optimizationPanelDock.toggleViewAction().triggered.connect(
            lambda checked, dock=self.optimizationPanelDock: self.animate_dock_toggle(
                dock, checked
            )
        )

    def animate_dock_toggle(self, dock_widget, show_state_after_toggle):
        # show_state_after_toggle is true if the action intends to make the dock visible

        # If the dock is part of a tab group, and it's not the visible one,
        # QDockWidget.isVisible() might be false even if its toggle action is checked.
        # We care about the state *after* the toggle action has conceptually occurred.

        is_currently_visible = (
            not dock_widget.isHidden()
        )  # More reliable for animation start

        # If the action wants to show it, and it's currently hidden -> animate in
        # If the action wants to hide it, and it's currently visible -> animate out

        animation_duration = 300  # ms
        easing_curve = QEasingCurve.InOutQuad

        original_width = self.dock_original_sizes.get(
            dock_widget, 300
        )  # Default if not found

        # Stop any existing animation for this dock
        if (
            dock_widget in self.dock_animations
            and self.dock_animations[dock_widget].state() == QPropertyAnimation.Running
        ):
            self.dock_animations[dock_widget].stop()

        if show_state_after_toggle:  # Animate IN
            if dock_widget.isHidden():  # Only animate if truly hidden
                dock_widget.show()  # Make it visible first
                # For tabbed docks, ensure it's raised
                if self.lensEditorDock.isTabbed() and dock_widget in [
                    self.lensEditorDock,
                    self.systemPropertiesDock,
                ]:
                    parent_tab_bar = (
                        self.lensEditorDock.parentWidget().parentWidget()
                    )  # Rough guess, might need QTabBar specific parent
                    if isinstance(parent_tab_bar, QTabWidget):
                        # Find index of dock_widget and set current
                        pass  # Complex logic for tab switching, defer for now

                # For left/right docks, animate width
                if self.dockWidgetArea(dock_widget) in [
                    Qt.DockWidgetArea.LeftDockWidgetArea,
                    Qt.DockWidgetArea.RightDockWidgetArea,
                ]:
                    animation = QPropertyAnimation(dock_widget, b"maximumWidth")
                    animation.setStartValue(0)
                    animation.setEndValue(original_width)
                else:  # For top/bottom docks, animate height
                    animation = QPropertyAnimation(dock_widget, b"maximumHeight")
                    animation.setStartValue(0)
                    animation.setEndValue(
                        dock_widget.sizeHint().height()
                        if dock_widget.sizeHint().height() > 0
                        else 200
                    )

                animation.setDuration(animation_duration)
                animation.setEasingCurve(easing_curve)
                animation.start(QPropertyAnimation.DeleteWhenStopped)
                self.dock_animations[dock_widget] = animation
            else:
                # If it's already visible (e.g. another tab in its group was visible), just ensure it's raised
                dock_widget.raise_()

        else:  # Animate OUT
            if not dock_widget.isHidden():  # Only animate if truly visible
                # For left/right docks
                if self.dockWidgetArea(dock_widget) in [
                    Qt.DockWidgetArea.LeftDockWidgetArea,
                    Qt.DockWidgetArea.RightDockWidgetArea,
                ]:
                    animation = QPropertyAnimation(dock_widget, b"maximumWidth")
                    animation.setStartValue(dock_widget.width())
                    animation.setEndValue(0)
                else:  # For top/bottom docks
                    animation = QPropertyAnimation(dock_widget, b"maximumHeight")
                    animation.setStartValue(dock_widget.height())
                    animation.setEndValue(0)

                animation.setDuration(animation_duration)
                animation.setEasingCurve(easing_curve)
                animation.finished.connect(dock_widget.hide)  # Hide when done
                # Restore maximumWidth/Height after hiding to allow normal resize if shown again without animation
                animation.finished.connect(
                    lambda: dock_widget.setMaximumWidth(original_width)
                    if self.dockWidgetArea(dock_widget)
                    in [
                        Qt.DockWidgetArea.LeftDockWidgetArea,
                        Qt.DockWidgetArea.RightDockWidgetArea,
                    ]
                    else dock_widget.setMaximumHeight(2000)
                )  # A large value
                animation.start(QPropertyAnimation.DeleteWhenStopped)
                self.dock_animations[dock_widget] = animation

    def _create_actions(self):
        # File actions
        self.newAction = QAction(
            "&New System",
            self,
            shortcut=QKeySequence.New,
            triggered=self.new_system_action,
        )
        self.openAction = QAction(
            "&Open System...",
            self,
            shortcut=QKeySequence.Open,
            triggered=self.open_system_action,
        )
        self.saveAction = QAction(
            "&Save System",
            self,
            shortcut=QKeySequence.Save,
            triggered=self.save_system_action,
        )
        self.saveAsAction = QAction(
            "Save System &As...",
            self,
            shortcut=QKeySequence.SaveAs,
            triggered=self.save_system_as_action,
        )
        self.exitAction = QAction(
            "E&xit", self, shortcut=QKeySequence.Quit, triggered=self.close
        )

        # View actions for toggling docks - text is set here, actual toggle connection in __init__
        self.toggleLensEditorAction = self.lensEditorDock.toggleViewAction()
        self.toggleLensEditorAction.setText("Toggle Lens &Editor")

        self.toggleSystemPropertiesAction = self.systemPropertiesDock.toggleViewAction()
        self.toggleSystemPropertiesAction.setText("Toggle System &Properties")

        self.toggleAnalysisPanelAction = self.analysisPanelDock.toggleViewAction()
        self.toggleAnalysisPanelAction.setText("Toggle &Analysis Panel")

        self.toggleOptimizationPanelAction = (
            self.optimizationPanelDock.toggleViewAction()
        )
        self.toggleOptimizationPanelAction.setText("Toggle &Optimization Panel")

        # Theme actions
        self.themeActionGroup = QActionGroup(self)
        self.themeActionGroup.setExclusive(True)

        self.darkThemeAction = QAction("Dark Theme", self, checkable=True)
        self.darkThemeAction.triggered.connect(
            lambda: self.switch_theme(THEME_DARK_PATH)
        )
        self.themeActionGroup.addAction(self.darkThemeAction)

        self.lightThemeAction = QAction("Light Theme", self, checkable=True)
        self.lightThemeAction.triggered.connect(
            lambda: self.switch_theme(THEME_LIGHT_PATH)
        )
        self.themeActionGroup.addAction(self.lightThemeAction)

        # Initialize check state after actions are created and current_theme_path is set
        if hasattr(self, "current_theme_path"):  # ensure current_theme_path exists
            if self.current_theme_path == THEME_DARK_PATH:
                self.darkThemeAction.setChecked(True)
            else:
                self.lightThemeAction.setChecked(True)

        # Help actions
        self.aboutAction = QAction(
            "&About Optiland GUI", self, triggered=self.about_action
        )

    def _create_menu_bar(self):
        menuBar = self.menuBar()

        fileMenu = menuBar.addMenu("&File")
        fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.saveAction)
        fileMenu.addAction(self.saveAsAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)

        viewMenu = menuBar.addMenu("&View")
        themeMenu = viewMenu.addMenu("&Theme")
        themeMenu.addAction(self.darkThemeAction)
        themeMenu.addAction(self.lightThemeAction)
        viewMenu.addSeparator()
        # Add the original toggle actions to the menu
        viewMenu.addAction(self.lensEditorDock.toggleViewAction())
        viewMenu.addAction(self.systemPropertiesDock.toggleViewAction())
        viewMenu.addAction(self.analysisPanelDock.toggleViewAction())
        viewMenu.addAction(self.optimizationPanelDock.toggleViewAction())

        runMenu = menuBar.addMenu("&Run")

        helpMenu = menuBar.addMenu("&Help")
        helpMenu.addAction(self.aboutAction)

    def _create_tool_bars(self):
        fileToolBar = self.addToolBar("File")
        fileToolBar.addAction(self.newAction)
        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.saveAction)

    def load_stylesheet(self, filepath):
        try:
            with open(filepath) as f:
                style_str = f.read()
                self.setStyleSheet(style_str)
                self.current_theme_path = filepath
                if hasattr(self, "darkThemeAction") and hasattr(
                    self, "lightThemeAction"
                ):
                    self.darkThemeAction.setChecked(filepath == THEME_DARK_PATH)
                    self.lightThemeAction.setChecked(filepath == THEME_LIGHT_PATH)
                print(f"Stylesheet loaded: {filepath}")
        except FileNotFoundError:
            print(f"Stylesheet not found: {filepath}")
            QMessageBox.warning(
                self,
                "Theme Error",
                f"Stylesheet not found: {os.path.basename(filepath)}",
            )
        except Exception as e:
            print(f"Error loading stylesheet: {e}")
            QMessageBox.critical(self, "Theme Error", f"Error loading stylesheet: {e}")

    @Slot(str)
    def switch_theme(self, theme_path):
        if theme_path != self.current_theme_path:
            self.load_stylesheet(theme_path)

    @Slot()
    def new_system_action(self):
        self.connector.new_system()
        print("Main Window: New System action triggered")

    @Slot()
    def open_system_action(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Optiland System",
            "",
            "Optiland JSON Files (*.json);;All Files (*)",
        )
        if filepath:
            self.connector.load_optic_from_file(filepath)
            print(f"Main Window: Open System action triggered - {filepath}")

    @Slot()
    def save_system_action(self):
        current_path = self.connector.get_current_filepath()
        if current_path:
            self.connector.save_optic_to_file(current_path)
            print(f"Main Window: Save System action triggered - {current_path}")
        else:
            self.save_system_as_action()

    @Slot()
    def save_system_as_action(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Optiland System As...",
            "",
            "Optiland JSON Files (*.json);;All Files (*)",
        )
        if filepath:
            if (
                not filepath.lower().endswith(".json")
                and "(*.json)" in _.split(";;")[0]
            ):
                filepath += ".json"
            self.connector.save_optic_to_file(filepath)
            print(f"Main Window: Save System As action triggered - {filepath}")

    @Slot()
    def about_action(self):
        # Create a custom QDialog for the About box to animate it
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("About Optiland GUI")
        layout = QVBoxLayout(about_dialog)
        about_text = QLabel(
            "<p><b>Optiland GUI</b></p>"
            "<p>A modern interface for the Optiland optical simulation package.</p>"
            "<p>Version: 0.1.4 (Animated About Dialog)</p>"
            "<p>Built with PySide6.</p>"
        )
        about_text.setTextFormat(Qt.TextFormat.RichText)  # Ensure HTML is rendered
        about_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(about_text)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(about_dialog.accept)
        layout.addWidget(ok_button, alignment=Qt.AlignmentFlag.AlignCenter)

        about_dialog.setLayout(layout)
        about_dialog.setMinimumSize(300, 200)  # Give it a reasonable default size

        # Fade-in animation
        about_dialog.setWindowOpacity(0.0)
        # Store animation on the dialog itself or on main window to prevent premature garbage collection if needed
        # For a short-lived dialog, direct storage might be okay if exec_ blocks until it's done.
        # However, if the animation object is collected, it stops.
        # Assigning to self.about_dialog_animation ensures it lives as long as MainWindow or until replaced.
        self.about_dialog_animation = QPropertyAnimation(about_dialog, b"windowOpacity")
        self.about_dialog_animation.setDuration(300)  # ms
        self.about_dialog_animation.setStartValue(0.0)
        self.about_dialog_animation.setEndValue(1.0)
        self.about_dialog_animation.setEasingCurve(QEasingCurve.InOutQuad)

        # Make sure the dialog is deleted after closing to free resources
        about_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # The animation should ideally be started just before or as the dialog becomes visible.
        # QDialog.exec_() is blocking. We can show it modelessly, animate, then make it modal,
        # or rely on Qt to handle the animation started just before exec_().
        # Starting animation then calling exec_()
        self.about_dialog_animation.start(
            QPropertyAnimation.DeletionPolicy.DeleteWhenStopped
        )
        about_dialog.exec_()

    def closeEvent(self, event):
        print("Main Window: Closing application.")
        event.accept()
