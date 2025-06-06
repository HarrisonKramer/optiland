
import inspect
import numpy as np 

from PySide6.QtCore import Slot, Qt, QSize
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QFrame,
    QSpacerItem,
    QSizePolicy,
    QGridLayout,
    QScrollArea,
    QFormLayout,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib 

from optiland.analysis import PupilAberration, RmsSpotSizeVsField, RmsWavefrontErrorVsField, ThroughFocusSpotDiagram, SpotDiagram, RayFan, YYbar, Distortion, GridDistortion, FieldCurvature, IncoherentIrradiance, EncircledEnergy
from optiland.mtf import FFTMTF, GeometricMTF

from .optiland_connector import OptilandConnector
from . import gui_plot_utils


class AnalysisPanel(QWidget):
    """
    A comprehensive panel for running and displaying various optical analyses.
    It features a selection of analysis types, a main display area for plots,
    and a collapsible settings panel to configure each analysis.
    """
    ANALYSIS_MAP = {
        "Spot Diagram": SpotDiagram,
        "Ray Fan": RayFan,
        "Distortion Plot": Distortion,
        "Grid Distortion": GridDistortion,
        "Field Curvature": FieldCurvature,
        "Encircled Energy": EncircledEnergy,
        "RMS Spot Size vs Field": RmsSpotSizeVsField,
        "RMS Wavefront Error vs Field": RmsWavefrontErrorVsField,
        "Through-Focus Spot Diagram": ThroughFocusSpotDiagram,
        #"Incoherent Irradiance": IncoherentIrradiance,
        "Pupil Aberration": PupilAberration,
        "Geometric MTF": GeometricMTF,
        "FFT MTF": FFTMTF,
        "YYbar": YYbar,
    }
    if not ANALYSIS_MAP:
        print("Warning: No analysis modules were successfully imported for AnalysisPanel.")

    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Analysis")
        self.setObjectName("AnalysisPanel")

        gui_plot_utils.apply_gui_matplotlib_styles()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.analysis_results_pages = []
        self.current_plot_page_index = -1
        self.active_mpl_canvas_widget = None
        self.active_mpl_toolbar_widget = None
        self.motion_notify_cid = None
        self.current_settings_widgets = {}

        # --- Top Bar for Analysis Selection and Control ---
        top_bar_layout = QHBoxLayout()
        top_bar_layout.addWidget(QLabel("Analysis Type:"))
        self.analysisTypeCombo = QComboBox()
        if self.ANALYSIS_MAP:
            self.analysisTypeCombo.addItems(list(self.ANALYSIS_MAP.keys()))
        self.analysisTypeCombo.setObjectName("AnalysisTypeCombo")
        top_bar_layout.addWidget(self.analysisTypeCombo)
        top_bar_layout.addSpacerItem(
            QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self.btnRun = QPushButton("Run")
        self.btnRun.setObjectName("RunAnalysisButton")
        self.btnRunAll = QPushButton("Run All")
        self.btnRunAll.setObjectName("RunAllAnalysisButton")
        self.btnStop = QPushButton("Stop")
        self.btnStop.setObjectName("StopAnalysisButton")
        top_bar_layout.addWidget(self.btnRun)
        top_bar_layout.addWidget(self.btnRunAll)
        top_bar_layout.addWidget(self.btnStop)
        main_layout.addLayout(top_bar_layout)

        # --- Main Content Area ---
        main_separator_line = QFrame()
        main_separator_line.setObjectName("MainSeparatorLine")
        main_separator_line.setFrameShape(QFrame.Shape.HLine)
        main_separator_line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(main_separator_line)

        main_content_layout = QHBoxLayout()
        main_content_layout.setSpacing(10)

        # --- Plot Display Frame (Left/Center) ---
        self.plot_display_frame = QFrame()
        self.plot_display_frame.setObjectName("PlotDisplayFrame")
        plot_display_frame_layout = QVBoxLayout(self.plot_display_frame)
        plot_display_frame_layout.setContentsMargins(5, 5, 5, 5)

        # Title bar for the plot area
        self.plot_area_title_bar_layout = QHBoxLayout()
        self.plotTitleLabel = QLabel("No Analysis Run")
        self.plotTitleLabel.setObjectName("PlotTitleLabel")
        self.plot_area_title_bar_layout.addWidget(self.plotTitleLabel)
        self.mpl_toolbar_in_titlebar_container = QWidget()
        self.mpl_toolbar_in_titlebar_container.setObjectName("MPLToolbarInTitlebarContainer")
        self.mpl_toolbar_in_titlebar_layout = QHBoxLayout(self.mpl_toolbar_in_titlebar_container)
        self.mpl_toolbar_in_titlebar_layout.setContentsMargins(0,0,0,0)
        self.plot_area_title_bar_layout.addWidget(self.mpl_toolbar_in_titlebar_container)
        self.mpl_toolbar_in_titlebar_container.setVisible(False)
        self.plot_area_title_bar_layout.addStretch()
        self.btnRefreshPlot = QPushButton("Refresh")
        self.btnRefreshPlot.setObjectName("RefreshPlotButton")
        self.plot_area_title_bar_layout.addWidget(self.btnRefreshPlot)
        self.toggleSettingsButton = QPushButton(">")
        self.toggleSettingsButton.setObjectName("ToggleSettingsButton")
        self.toggleSettingsButton.setFixedSize(25, 25)
        self.plot_area_title_bar_layout.addWidget(self.toggleSettingsButton)
        plot_display_frame_layout.addLayout(self.plot_area_title_bar_layout)
        
        # Separator line
        title_plot_separator_line = QFrame()
        title_plot_separator_line.setFrameShape(QFrame.Shape.HLine)
        plot_display_frame_layout.addWidget(title_plot_separator_line)

        # Plot content and pagination
        plot_content_and_pages_layout = QHBoxLayout()
        plot_content_and_pages_layout.setContentsMargins(0,0,0,0)
        plot_and_info_widget = QWidget()
        plot_and_info_layout = QVBoxLayout(plot_and_info_widget)
        plot_and_info_layout.setContentsMargins(0,0,0,0)
        self.plot_container_widget = QWidget()
        self.plot_container_widget.setObjectName("PlotContainerWidget")
        self.plot_container_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot_and_info_layout.addWidget(self.plot_container_widget, 1)
        self.cursor_coord_label = QLabel("", self.plot_container_widget)
        self.cursor_coord_label.setObjectName("CursorCoordLabel")
        self.cursor_coord_label.setStyleSheet("background-color:rgba(0,0,0,0.65);color:white;padding:2px 4px;border-radius:3px;")
        self.cursor_coord_label.setVisible(False)
        self.cursor_coord_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.dataInfoLabel = QLabel("Data Analysis info will appear here.")
        self.dataInfoLabel.setObjectName("DataInfoLabel")
        plot_and_info_layout.addWidget(self.dataInfoLabel)
        plot_content_and_pages_layout.addWidget(plot_and_info_widget, 1)
        self.page_buttons_scroll_area = QScrollArea()
        self.page_buttons_scroll_area.setObjectName("PageButtonsScrollArea")
        self.page_buttons_scroll_area.setWidgetResizable(True)
        self.page_buttons_scroll_area.setFixedWidth(30)
        page_buttons_container_widget = QWidget()
        self.vertical_page_buttons_layout = QVBoxLayout(page_buttons_container_widget)
        self.vertical_page_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.page_buttons_scroll_area.setWidget(page_buttons_container_widget)
        plot_content_and_pages_layout.addWidget(self.page_buttons_scroll_area)
        plot_display_frame_layout.addLayout(plot_content_and_pages_layout, 1)
        main_content_layout.addWidget(self.plot_display_frame, 3)

        # --- Settings Panel (Right) ---
        self.settings_area_widget = QWidget()
        self.settings_area_widget.setObjectName("SettingsArea")
        self.settings_area_widget.setFixedWidth(250)
        settings_layout = QVBoxLayout(self.settings_area_widget)
        settings_layout.setContentsMargins(5,5,5,5)
        settings_layout.addWidget(QLabel("Analysis Settings"))
        self.settings_scroll_area = QScrollArea()
        self.settings_scroll_area.setWidgetResizable(True)
        self.settingsContentWidget = QWidget()
        self.settings_form_layout = QFormLayout(self.settingsContentWidget)
        self.settings_scroll_area.setWidget(self.settingsContentWidget)
        settings_layout.addWidget(self.settings_scroll_area, 1)
        self.btnApplySettingsAndRerun = QPushButton("Apply Settings & Rerun")
        self.btnApplySettingsAndRerun.setObjectName("ApplySettingsAndRerunButton")
        settings_layout.addWidget(self.btnApplySettingsAndRerun)
        self.btnApplySettingsAndRerun.setVisible(False)
        main_content_layout.addWidget(self.settings_area_widget, 1)
        main_layout.addLayout(main_content_layout, 1)

        # --- Log Area (Bottom) ---
        self.logArea = QTextEdit()
        self.logArea.setObjectName("LogArea")
        self.logArea.setReadOnly(True)
        self.logArea.setFixedHeight(60)
        main_layout.addWidget(self.logArea)

        # --- Connections ---
        self.btnRun.clicked.connect(self.run_analysis_slot)
        self.btnRunAll.clicked.connect(self.run_all_analysis_slot)
        self.btnStop.clicked.connect(self.stop_analysis_slot)
        self.analysisTypeCombo.currentTextChanged.connect(self.on_analysis_type_changed)
        self.toggleSettingsButton.clicked.connect(self.toggle_settings_panel_slot)
        self.btnRefreshPlot.clicked.connect(self._refresh_current_plot_page_slot)
        self.btnApplySettingsAndRerun.clicked.connect(self._apply_settings_and_rerun_analysis_slot)

        # --- Initial State ---
        self.on_analysis_type_changed(self.analysisTypeCombo.currentText())
        self.update_pagination_ui()
        self.display_plot_page(self.current_plot_page_index)
        self.settings_area_widget.setVisible(False)
        self.toggleSettingsButton.setText(">")

    def _clear_layout(self, layout_to_clear):
        if layout_to_clear is not None:
            while layout_to_clear.count():
                item = layout_to_clear.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    if isinstance(widget, FigureCanvas):
                        if hasattr(widget, '_motion_notify_cid') and widget._motion_notify_cid is not None:
                            try: widget.mpl_disconnect(widget._motion_notify_cid)
                            except TypeError: pass
                            widget._motion_notify_cid = None
                        plt.close(widget.figure)
                    widget.setParent(None); widget.deleteLater()
                else:
                    sub_layout = item.layout()
                    if sub_layout is not None: self._clear_layout(sub_layout)

    def _add_setting_widget(self, param_name, param_info, default_value_override=None):
        """Helper to add a settings widget based on param_info."""
        label_text = param_name.replace("_", " ").title() + ":"
        default_value = param_info.get("default") if default_value_override is None else default_value_override
        annotation = param_info.get("annotation")
        widget = None

        if annotation is inspect.Parameter.empty or annotation is None:
            if isinstance(default_value, bool): annotation = bool
            elif isinstance(default_value, int): annotation = int
            elif isinstance(default_value, float): annotation = float
            elif isinstance(default_value, str): annotation = str

        if annotation == int:
            widget = QSpinBox()
            min_v, max_v, step_v = -1000000, 1000000, 1
            if param_name in ["num_rays", "num_points"]: min_v, max_v = 1, 10000000
            elif param_name in ["num_rings", "num_fields"]: min_v, max_v = 1, 1024
            elif param_name == "num_steps": min_v, max_v = 1, 51
            elif param_name == "detector_surface": min_v, max_v = -100, 100
            widget.setRange(min_v, max_v)
            widget.setSingleStep(step_v)
            if param_name == "num_steps" and default_value and default_value % 2 == 0: default_value += 1 # Ensure odd
            widget.setValue(int(default_value) if default_value is not None else 0)

        elif annotation == float:
            widget = QDoubleSpinBox()
            widget.setDecimals(4)
            widget.setRange(-1e9, 1e9)
            widget.setSingleStep(0.01 if "delta_focus" in param_name else 0.1)
            widget.setValue(float(default_value) if default_value is not None else 0.0)

        elif annotation == bool:
            widget = QCheckBox(param_name.replace("_", " ").title())
            widget.setChecked(bool(default_value) if default_value is not None else False)
            label_text = ""

        elif "Literal" in str(annotation):
            from typing import get_args
            options = get_args(annotation)
            if options:
                widget = QComboBox(); widget.addItems([str(opt) for opt in options])
                if str(default_value) in [str(o) for o in options]: widget.setCurrentText(str(default_value))

        elif annotation == str:
            if param_name == "distribution":
                widget = QComboBox(); widget.addItems(["hexapolar", "grid", "random", "ring", "line_x", "line_y", "gaussian", "uniform"])
            elif param_name in ["coordinates", "distortion_type"]:
                widget = QComboBox(); widget.addItems(["f-tan", "f-theta"] if "distortion" in param_name else ["local", "global"])
            elif param_name == "cmap":
                widget = QComboBox(); widget.addItems(["inferno", "viridis", "plasma", "magma", "gray", "jet"])
            else: # Fallback for other strings (like wavelength='primary')
                widget = QLineEdit()
            if isinstance(widget, QComboBox): widget.setCurrentText(str(default_value) if default_value else widget.itemText(0))
            else: widget.setText(str(default_value) if default_value is not None else "")

        elif annotation == tuple or isinstance(default_value, tuple):
            widget = QLineEdit(",".join(map(str, default_value)) if default_value else "")
            widget.setPlaceholderText("e.g., 128,128")

        if widget:
            if isinstance(widget, QCheckBox): self.settings_form_layout.addRow(widget)
            else: self.settings_form_layout.addRow(QLabel(label_text), widget)
            self.current_settings_widgets[param_name] = widget
        else:
            print(f"Warning: No widget for '{param_name}' (annotation: {annotation})")

    def _update_settings_ui(self, analysis_name: str):
        while self.settings_form_layout.rowCount() > 0:
            self.settings_form_layout.removeRow(0)
        self.current_settings_widgets.clear()

        analysis_class = self.ANALYSIS_MAP.get(analysis_name)
        if not analysis_class:
            self.settings_form_layout.addRow(QLabel("No settings available."))
            return

        init_params = gui_plot_utils.get_analysis_parameters(analysis_class)
        # Known view args to add controls for if not in constructor
        view_arg_defaults = {
            "add_airy_disk": (bool, False),
            "cmap": (str, "inferno"),
            "normalize": (bool, True),
            "cross_section": (str, ""),
        }

        # Add constructor params
        for param_name, param_info in init_params.items():
            self._add_setting_widget(param_name, param_info)

        # Add controls for known view args if the analysis class has 'view' and the arg is not already in __init__
        if hasattr(analysis_class, 'view') and callable(analysis_class.view):
            view_sig = inspect.signature(analysis_class.view)
            for view_arg, (v_type, v_default) in view_arg_defaults.items():
                if view_arg in view_sig.parameters and view_arg not in init_params:
                     self._add_setting_widget(view_arg, {"default": v_default, "annotation": v_type})

        self.settings_form_layout.addItem(QSpacerItem(20, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

    @Slot(str)
    def on_analysis_type_changed(self, analysis_name: str):
        self._update_settings_ui(analysis_name)
        if self.current_plot_page_index == -1 or not self.analysis_results_pages:
             self.plotTitleLabel.setText(analysis_name)

    def update_pagination_ui(self):
        self._clear_layout(self.vertical_page_buttons_layout)
        for i, page_data in enumerate(self.analysis_results_pages):
            btn_page = QPushButton(str(i + 1))
            btn_page.setObjectName(f"PageButton_{i+1}")
            btn_page.setCheckable(True)
            btn_page.setChecked(i == self.current_plot_page_index)
            btn_page.clicked.connect(lambda checked=False, index=i: self.switch_plot_page(index))
            self.vertical_page_buttons_layout.addWidget(btn_page)
        self.vertical_page_buttons_layout.addStretch()

    def switch_plot_page(self, page_index):
        if 0 <= page_index < len(self.analysis_results_pages):
            self.current_plot_page_index = page_index
            self.update_pagination_ui()
            self.display_plot_page(page_index)
            page_data = self.analysis_results_pages[page_index]
            self.logArea.append(f"Switched to page {page_index + 1}: {page_data.get('name', 'Analysis')}")
        else:
            self.current_plot_page_index = -1
            self.update_pagination_ui()
            self.display_plot_page(-1)
            self._update_settings_ui(self.analysisTypeCombo.currentText())

    def on_mouse_move_on_plot(self, event):
        if event.inaxes and self.active_mpl_canvas_widget:
            x_coord = f"{event.xdata:.2f}" if event.xdata is not None else "---"
            y_coord = f"{event.ydata:.2f}" if event.ydata is not None else "---"
            self.cursor_coord_label.setText(f"(x, y) = ({x_coord}, {y_coord})")
            self.cursor_coord_label.adjustSize()
            self.cursor_coord_label.move(5, 5)
            self.cursor_coord_label.setVisible(True)
            self.cursor_coord_label.raise_()
        elif self.active_mpl_canvas_widget:
            self.cursor_coord_label.setVisible(False)

    def display_plot_page(self, page_index):
        if self.active_mpl_toolbar_widget:
            self.mpl_toolbar_in_titlebar_layout.removeWidget(self.active_mpl_toolbar_widget)
            self.active_mpl_toolbar_widget.deleteLater(); self.active_mpl_toolbar_widget = None
        self.mpl_toolbar_in_titlebar_container.setVisible(False)

        if self.active_mpl_canvas_widget and self.motion_notify_cid:
            try: self.active_mpl_canvas_widget.mpl_disconnect(self.motion_notify_cid)
            except TypeError: pass
            self.motion_notify_cid = None
        self.cursor_coord_label.setVisible(False)

        plot_content_area_layout = self.plot_container_widget.layout()
        if plot_content_area_layout: self._clear_layout(plot_content_area_layout)
        else: plot_content_area_layout = QVBoxLayout(self.plot_container_widget)
        self.plot_container_widget.setLayout(plot_content_area_layout)

        self.active_mpl_canvas_widget = None

        if 0 <= page_index < len(self.analysis_results_pages):
            page_data = self.analysis_results_pages[page_index]
            analysis_name = page_data.get("name", "Analysis")
            self.plotTitleLabel.setText(analysis_name)
            self.dataInfoLabel.setText(page_data.get("result_summary", f"Results for {analysis_name}"))

            self._update_settings_ui(analysis_name)
            page_args = {**page_data.get("constructor_args_used", {}), **page_data.get("view_args", {})}
            for param_name, widget in self.current_settings_widgets.items():
                if param_name in page_args:
                    val = page_args[param_name]
                    if isinstance(widget, (QSpinBox, QDoubleSpinBox)): widget.setValue(val)
                    elif isinstance(widget, QComboBox): widget.setCurrentText(str(val))
                    elif isinstance(widget, QCheckBox): widget.setChecked(bool(val))
                    elif isinstance(widget, QLineEdit): widget.setText(",".join(map(str,val)) if isinstance(val,tuple) else str(val))

            analysis_instance = page_data.get("analysis_instance")
            if page_data.get("plot_type") == "embedded_mpl" and analysis_instance:
                fig = Figure(figsize=page_data.get("figsize", (7,5)), dpi=100)
                canvas = FigureCanvas(fig)
                self.active_mpl_canvas_widget = canvas
                analysis_instance.view(fig_to_plot_on=fig, **page_data.get("view_args", {}))

                self.active_mpl_toolbar_widget = NavigationToolbar(canvas, self.mpl_toolbar_in_titlebar_container)
                self.mpl_toolbar_in_titlebar_layout.addWidget(self.active_mpl_toolbar_widget)
                self.mpl_toolbar_in_titlebar_container.setVisible(True)

                plot_content_area_layout.addWidget(canvas)
                self.motion_notify_cid = canvas.mpl_connect('motion_notify_event', self.on_mouse_move_on_plot)
                canvas._motion_notify_cid = self.motion_notify_cid
            else:
                plot_content_area_layout.addWidget(QLabel(f"Cannot embed plot for {analysis_name}"))
        else:
            self.plotTitleLabel.setText("No Analysis Selected")
            self.dataInfoLabel.setText("Run an analysis to see results.")
            plot_content_area_layout.addWidget(QLabel("Select or Run an Analysis"))
            self._update_settings_ui(self.analysisTypeCombo.currentText())

    @Slot()
    def toggle_settings_panel_slot(self):
        is_visible = self.settings_area_widget.isVisible()
        self.settings_area_widget.setVisible(not is_visible)
        self.btnApplySettingsAndRerun.setVisible(not is_visible)
        self.toggleSettingsButton.setText("<" if not is_visible else ">")
        if not is_visible: # if we just made it visible
            self.display_plot_page(self.current_plot_page_index) # Re-run display logic to ensure settings are populated

    def _parse_tuple_str(self, s, expected_type=float, expected_len=2):
        if not s or not isinstance(s, str): return None
        try:
            parts = tuple(map(expected_type, s.split(',')))
            return parts if len(parts) == expected_len else None
        except (ValueError, TypeError): return None

    def _collect_current_settings(self):
        constructor_args, view_args = {}, {}
        for param_name, widget in self.current_settings_widgets.items():
            value = None
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)): value = widget.value()
            elif isinstance(widget, QComboBox): value = widget.currentText()
            elif isinstance(widget, QCheckBox): value = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if param_name in ["fields", "wavelengths", "wavelength"]: value = text if text else None
                elif param_name == "res": value = self._parse_tuple_str(text, int, 2)
                elif param_name == "px_size": value = self._parse_tuple_str(text, float, 2)
                elif param_name == "cross_section":
                    if not text: value = None
                    else:
                        parts = [p.strip() for p in text.split(',')]
                        if len(parts) == 2 and parts[0].lower() in ["cross-x", "cross-y"]:
                            try: value = (parts[0].lower(), int(parts[1]))
                            except ValueError: value = text
                        else: value = text
                else: value = text
            if value is not None:
                if param_name in ["add_airy_disk", "cmap", "normalize", "cross_section"]: view_args[param_name] = value
                else: constructor_args[param_name] = value
        return constructor_args, view_args

    def _execute_analysis(self, analysis_class, analysis_name):
        optic = self.connector.get_optic()
        if not optic or optic.surface_group.num_surfaces < 2:
            QMessageBox.warning(self, "Analysis Error", "Minimal system required.")
            return None
        if optic.wavelengths.num_wavelengths == 0:
            QMessageBox.warning(self, "Analysis Error", "Optic has no wavelengths.")
            return None
        
        try:
            constructor_args, view_args = self._collect_current_settings()
            final_args = {"optic": optic}
            valid_init = gui_plot_utils.get_analysis_parameters(analysis_class).keys()
            for k,v in constructor_args.items():
                if k in valid_init: final_args[k] = v

            print(f"LOG: Executing {analysis_name} with args: {final_args}")
            instance = analysis_class(**final_args)
            
            can_embed = hasattr(instance, 'view') and 'fig_to_plot_on' in inspect.signature(instance.view).parameters
            if not can_embed: instance.view(**view_args) # External view
            
            page_data = {
                "name": analysis_name, "analysis_instance": instance,
                "plot_type": "embedded_mpl" if can_embed else "external_window",
                "view_args": view_args, "constructor_args_used": {k:v for k,v in final_args.items() if k!='optic'}
            }
            # Add dynamic figsize logic here if needed
            if analysis_name == "Through-Focus Spot Diagram":
                num_f = len(optic.fields.get_field_coords())
                num_s = final_args.get("num_steps", 5)
                page_data["figsize"] = (max(1,num_s) * 3, max(1,num_f) * 3)

            return page_data
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Error during {analysis_name}:\n{e}")
            import traceback; print(f"Analysis Panel Error: {e}\n{traceback.format_exc()}")
            return None

    @Slot()
    def _apply_settings_and_rerun_analysis_slot(self):
        if not (0 <= self.current_plot_page_index < len(self.analysis_results_pages)):
            return
        page_data = self.analysis_results_pages[self.current_plot_page_index]
        analysis_name = page_data.get("name")
        self.logArea.setText(f"Rerunning {analysis_name} with new settings...")
        new_page_data = self._execute_analysis(self.ANALYSIS_MAP[analysis_name], analysis_name)
        if new_page_data:
            self.analysis_results_pages[self.current_plot_page_index] = new_page_data
            self.display_plot_page(self.current_plot_page_index)
            self.logArea.append(f"{analysis_name} reran successfully.")

    @Slot()
    def _refresh_current_plot_page_slot(self):
        if not (0 <= self.current_plot_page_index < len(self.analysis_results_pages)): return
        self.logArea.setText(f"Refreshing plot...")
        self.display_plot_page(self.current_plot_page_index)
        self.logArea.append(f"Plot refreshed.")

    @Slot()
    def run_analysis_slot(self):
        analysis_name = self.analysisTypeCombo.currentText()
        analysis_class = self.ANALYSIS_MAP.get(analysis_name)
        if not analysis_class: return
        self.logArea.setText(f"Running {analysis_name}...")
        page_data = self._execute_analysis(analysis_class, analysis_name)
        if page_data:
            self.analysis_results_pages.append(page_data)
            self.switch_plot_page(len(self.analysis_results_pages) - 1)
            self.logArea.append(f"{analysis_name} run complete.")

    @Slot()
    def run_all_analysis_slot(self): self.logArea.append("Run All: Not yet implemented.")
    @Slot()
    def stop_analysis_slot(self): self.logArea.append("Stop: Not yet implemented.")
