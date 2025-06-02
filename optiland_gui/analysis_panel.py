# optiland_gui/analysis_panel.py
import inspect

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
    QFormLayout
)

from optiland.analysis import (
    Distortion,
    EncircledEnergy,
    FieldCurvature,
    GridDistortion,
    PupilAberration,
    RayFan,
    RmsSpotSizeVsField,
    RmsWavefrontErrorVsField,
    SpotDiagram,
    YYbar,
)
from optiland.mtf import FFTMTF, GeometricMTF

from .optiland_connector import OptilandConnector


class AnalysisPanel(QWidget):
    ANALYSIS_MAP = {
        "Spot Diagram": SpotDiagram,
        "Encircled Energy": EncircledEnergy,
        "Ray Fan": RayFan,
        "Y-Ybar Diagram": YYbar,
        "Distortion Plot": Distortion,
        "Grid Distortion": GridDistortion,
        "Field Curvature": FieldCurvature,
        "RMS Spot Size vs Field": RmsSpotSizeVsField,
        "RMS Wavefront Error vs Field": RmsWavefrontErrorVsField,
        "Pupil Aberration": PupilAberration,
        "Geometric MTF": GeometricMTF,
        "FFT MTF": FFTMTF,
    }

    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Analysis")

        self.setObjectName("AnalysisPanel")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.analysis_results_pages = []  # To store data for each page
        self.current_plot_page_index = -1 # No page selected initially

        # --- Top Bar ---
        top_bar_layout = QHBoxLayout()
        top_bar_layout.addWidget(QLabel("Analysis Type:"))
        self.analysisTypeCombo = QComboBox()
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

        # --- Horizontal Separator Line ---
        separator_line = QFrame()
        separator_line.setObjectName("SeparatorLine")
        separator_line.setFrameShape(QFrame.Shape.HLine)
        separator_line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator_line)

        # --- Main Content Area (Plot + Settings) ---
        main_content_layout = QHBoxLayout()
        main_content_layout.setSpacing(10)

        # Left/Central Area (Plot Display Frame)
        self.plot_display_frame = QFrame()
        self.plot_display_frame.setObjectName("PlotDisplayFrame")
        self.plot_display_frame.setFrameShape(QFrame.Shape.StyledPanel) # Reverted for dark theme preference
        self.plot_display_frame.setFrameShadow(QFrame.Shadow.Plain)    # Reverted for dark theme preference
        # self.plot_display_frame.setLineWidth(1) # lineWidth usually 0 for Plain shadow, 1 for others
        plot_display_frame_layout = QVBoxLayout(self.plot_display_frame) # Main layout for this frame
        plot_display_frame_layout.setContentsMargins(5, 5, 5, 5)

        # Title bar for the plot area (with toggle button)
        plot_area_title_bar_layout = QHBoxLayout()
        self.plotTitleLabel = QLabel("No Analysis Run")
        self.plotTitleLabel.setObjectName("PlotTitleLabel")
        plot_area_title_bar_layout.addWidget(self.plotTitleLabel)
        plot_area_title_bar_layout.addStretch()
        self.toggleSettingsButton = QPushButton(">")
        self.toggleSettingsButton.setObjectName("ToggleSettingsButton")
        self.toggleSettingsButton.setFixedSize(30, 30)
        self.toggleSettingsButton.setToolTip("Toggle Settings Panel")
        plot_area_title_bar_layout.addWidget(self.toggleSettingsButton)
        plot_display_frame_layout.addLayout(plot_area_title_bar_layout)

        # Content within the plot display frame (plots on left, vertical pages on right)
        plot_content_and_pages_layout = QHBoxLayout()
        plot_content_and_pages_layout.setContentsMargins(0,0,0,0)
        plot_content_and_pages_layout.setSpacing(5)

        # Area for plot placeholders and data info label
        plot_and_info_widget = QWidget()
        plot_and_info_layout = QVBoxLayout(plot_and_info_widget)
        plot_and_info_layout.setContentsMargins(0,0,0,0)

        self.plotAreaWidget = QWidget()
        self.plotAreaWidget.setObjectName("PlotAreaWidget")
        self.plotAreaWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_area_grid_layout = QGridLayout(self.plotAreaWidget) # Made instance member
        self.plot_area_grid_layout.setContentsMargins(0,0,0,0)

        self.plotPlaceholder1 = QLabel("Plot Area 1")
        self.plotPlaceholder1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plotPlaceholder1.setFrameShape(QFrame.Shape.StyledPanel)
        self.plotPlaceholder1.setMinimumSize(150, 150) # Adjusted min size
        self.plot_area_grid_layout.addWidget(self.plotPlaceholder1, 0, 0)

        self.plotPlaceholder2 = QLabel("Plot Area 2")
        self.plotPlaceholder2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plotPlaceholder2.setFrameShape(QFrame.Shape.StyledPanel)
        self.plotPlaceholder2.setMinimumSize(150, 150) # Adjusted min size
        self.plot_area_grid_layout.addWidget(self.plotPlaceholder2, 0, 1)

        self.plotPlaceholder3 = QLabel("Plot Area 3")
        self.plotPlaceholder3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plotPlaceholder3.setFrameShape(QFrame.Shape.StyledPanel)
        self.plotPlaceholder3.setMinimumSize(150, 150) # Adjusted min size
        self.plot_area_grid_layout.addWidget(self.plotPlaceholder3, 0, 2)
        plot_and_info_layout.addWidget(self.plotAreaWidget, 1)

        self.dataInfoLabel = QLabel("Data Analysis info will appear here.")
        self.dataInfoLabel.setObjectName("DataInfoLabel")
        plot_and_info_layout.addWidget(self.dataInfoLabel)
        plot_content_and_pages_layout.addWidget(plot_and_info_widget, 1) # Main plot area takes stretch

        # Vertical Scrollable Pagination Buttons Area
        self.page_buttons_scroll_area = QScrollArea()
        self.page_buttons_scroll_area.setObjectName("PageButtonsScrollArea")
        self.page_buttons_scroll_area.setWidgetResizable(True)
        self.page_buttons_scroll_area.setFixedWidth(30) # Adjust width as needed for buttons + scrollbar
        self.page_buttons_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.page_buttons_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        page_buttons_container_widget = QWidget() # Widget to hold the layout for scroll area
        self.vertical_page_buttons_layout = QVBoxLayout(page_buttons_container_widget)
        self.vertical_page_buttons_layout.setContentsMargins(2, 2, 2, 2)
        self.vertical_page_buttons_layout.setSpacing(5)
        self.vertical_page_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Buttons align to top

        self.page_buttons_scroll_area.setWidget(page_buttons_container_widget)
        plot_content_and_pages_layout.addWidget(self.page_buttons_scroll_area)

        plot_display_frame_layout.addLayout(plot_content_and_pages_layout, 1) # Add combined layout to frame

        main_content_layout.addWidget(self.plot_display_frame, 3) # Plot frame takes more space

        # Right Settings Area
        self.settings_area_widget = QWidget()
        self.settings_area_widget.setObjectName("SettingsArea")
        self.settings_area_widget.setFixedWidth(250)
        settings_layout = QVBoxLayout(self.settings_area_widget)
        settings_layout.setContentsMargins(5,5,5,5)

        settings_title_label = QLabel("Settings here")
        settings_title_label.setObjectName("SettingsTitleLabel")
        settings_layout.addWidget(settings_title_label)

        scroll_area_settings = QScrollArea() # Renamed to avoid conflict
        scroll_area_settings.setObjectName("SettingsScrollArea")
        scroll_area_settings.setWidgetResizable(True)
        scroll_area_settings.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.settingsContentWidget = QWidget()
        self.settingsContentWidget.setObjectName("SettingsContentWidget")
        settings_form_layout = QFormLayout(self.settingsContentWidget)
        settings_form_layout.setContentsMargins(10,10,10,10)
        settings_form_layout.setSpacing(8)
        settings_form_layout.addRow("Setting Option A:", QLabel("Value"))
        settings_form_layout.addRow("Setting Option B:", QLabel("Another Value"))
        settings_form_layout.addRow(QLabel("More settings can go here,\nand this area will scroll if needed."))
        settings_form_layout.addRow("Parameter X:", QLabel("True"))
        settings_form_layout.addRow("Parameter Y:", QLabel("123.45"))
        settings_form_layout.addRow("Wavelength:", QLabel("0.550 µm"))
        settings_form_layout.addRow("Number of Rays:", QLabel("1000"))
        settings_form_layout.addRow("Show Airy Disk:", QLabel("Yes"))

        scroll_area_settings.setWidget(self.settingsContentWidget)
        settings_layout.addWidget(scroll_area_settings, 1)

        main_content_layout.addWidget(self.settings_area_widget, 1)
        main_layout.addLayout(main_content_layout, 1)

        # --- Bottom Log Area ---
        self.logArea = QTextEdit()
        self.logArea.setObjectName("LogArea")
        self.logArea.setReadOnly(True)
        self.logArea.setPlaceholderText(
            "Select an analysis and click 'Run'. Plots will appear in external windows for now."
        )
        self.logArea.setFixedHeight(80)
        main_layout.addWidget(self.logArea)

        # Connect signals
        self.btnRun.clicked.connect(self.run_analysis_slot)
        self.btnRunAll.clicked.connect(self.run_all_analysis_slot)
        self.btnStop.clicked.connect(self.stop_analysis_slot)
        self.analysisTypeCombo.currentTextChanged.connect(self.update_ui_for_analysis_type)
        self.toggleSettingsButton.clicked.connect(self.toggle_settings_panel_slot)

        # Initialize
        self.update_ui_for_analysis_type(self.analysisTypeCombo.currentText())
        self.update_pagination_ui() # Initial call to set up pagination (will be empty)
        self.display_plot_page(self.current_plot_page_index) # Initial placeholder display
        
        self.settings_area_widget.setVisible(False) # Start with settings panel hidden
        self.toggleSettingsButton.setText(">")  


    def _clear_layout(self, layout):
        """Helper to remove all widgets from a layout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else: # If it's a sub-layout
                    sub_layout = item.layout()
                    if sub_layout is not None:
                        self._clear_layout(sub_layout)

    def update_pagination_ui(self):
        self._clear_layout(self.vertical_page_buttons_layout)

        for i, page_data in enumerate(self.analysis_results_pages):
            btn_page = QPushButton(str(i + 1))
            btn_page.setObjectName(f"PageButton_{i+1}")
            #btn_page.setFixedSize(15, 15)
            btn_page.setCheckable(True)
            btn_page.setChecked(i == self.current_plot_page_index)
            btn_page.clicked.connect(lambda checked=False, index=i: self.switch_plot_page(index))
            self.vertical_page_buttons_layout.addWidget(btn_page)
        # self.vertical_page_buttons_layout.addStretch() # Pushes buttons to top

    def switch_plot_page(self, page_index):
        if 0 <= page_index < len(self.analysis_results_pages):
            self.current_plot_page_index = page_index
            self.update_pagination_ui() # Update button styles
            self.display_plot_page(page_index)
        else:
            # Handle case where page_index is out of bounds, maybe display default
            self.display_plot_page(-1) # Display default/empty state

    def display_plot_page(self, page_index):
        if 0 <= page_index < len(self.analysis_results_pages):
            page_data = self.analysis_results_pages[page_index]
            analysis_name = page_data.get("name", "Analysis")
            # constructor_args = page_data.get("args", {}) # For future use
            # analysis_instance = page_data.get("instance", None) # For future use

            self.plotTitleLabel.setText(analysis_name)
            self.dataInfoLabel.setText(f"Displaying results for {analysis_name} (Page {page_index + 1})")
            self.logArea.append(f"Switched to page {page_index + 1}: {analysis_name}")

            # Update plot placeholders based on analysis type
            if analysis_name == "Spot Diagram":
                self.plotPlaceholder1.setText(f"Page {page_index+1}\nHx: 0.000, Hy: 0.000")
                self.plotPlaceholder2.setText(f"Page {page_index+1}\nHx: 0.000, Hy: 0.700")
                self.plotPlaceholder3.setText(f"Page {page_index+1}\nHx: 0.000, Hy: 1.000")
                self.plotPlaceholder1.setVisible(True)
                self.plotPlaceholder2.setVisible(True)
                self.plotPlaceholder3.setVisible(True)
            elif analysis_name == "Pupil Aberration":
                self.plotPlaceholder1.setText(f"Page {page_index+1}\nPupil Aberration Data")
                self.plotPlaceholder1.setVisible(True)
                self.plotPlaceholder2.setVisible(False)
                self.plotPlaceholder3.setVisible(False)
            else:
                self.plotPlaceholder1.setText(f"Page {page_index+1}\n{analysis_name} Data")
                self.plotPlaceholder1.setVisible(True)
                self.plotPlaceholder2.setVisible(False)
                self.plotPlaceholder3.setVisible(False)
        else: # Default/empty state
            self.plotTitleLabel.setText("No Analysis Selected")
            self.dataInfoLabel.setText("Run an analysis to see results.")
            self.plotPlaceholder1.setText("Plot Area 1")
            self.plotPlaceholder2.setText("Plot Area 2")
            self.plotPlaceholder3.setText("Plot Area 3")
            self.plotPlaceholder1.setVisible(True)
            self.plotPlaceholder2.setVisible(True)
            self.plotPlaceholder3.setVisible(True)


    @Slot(str)
    def update_ui_for_analysis_type(self, analysis_name):
        # This slot is primarily for when the user *changes* the combo box.
        # It doesn't necessarily mean a new page is created yet.
        # We can update the main plot title to reflect current selection,
        # but the actual page content is managed by display_plot_page.
        if self.current_plot_page_index == -1 or not self.analysis_results_pages : # If no active page, update title
             self.plotTitleLabel.setText(analysis_name)
        # Current page content should reflect the selected page, not necessarily the combo box if pages exist
        # self.logArea.append(f"Analysis type selected: {analysis_name}")


    @Slot()
    def toggle_settings_panel_slot(self):
        if self.settings_area_widget.isVisible():
            self.settings_area_widget.setVisible(False)
            self.toggleSettingsButton.setText(">")
        else:
            self.settings_area_widget.setVisible(True)
            self.toggleSettingsButton.setText("<")


    @Slot()
    def run_analysis_slot(self):
        selected_analysis_name = self.analysisTypeCombo.currentText()
        analysis_class = self.ANALYSIS_MAP.get(selected_analysis_name)
        optic = self.connector.get_optic()

        if not optic or optic.surface_group.num_surfaces <= 2:
            QMessageBox.warning(self, "Analysis Error", "Cannot run analysis on an empty or minimal system.")
            self.logArea.setText("Cannot run analysis: Optic system is not sufficiently defined.")
            return
        if optic.wavelengths.num_wavelengths == 0:
            QMessageBox.warning(self, "Analysis Error", "Optic has no wavelengths defined.")
            self.logArea.setText("Cannot run analysis: No wavelengths defined in the optic.")
            return

        if analysis_class:
            self.logArea.setText(f"Running {selected_analysis_name}...")
            try:
                primary_wl_val = self.connector._get_safe_primary_wavelength_value()
                sig = inspect.signature(analysis_class.__init__)
                constructor_args = {}
                for param in sig.parameters.values():
                    if param.name == "self": continue
                    if param.name in ["optic", "optical_system"]: constructor_args[param.name] = optic
                    elif param.name == "fields": constructor_args[param.name] = "all"
                    elif param.name == "wavelengths":
                        constructor_args[param.name] = [primary_wl_val] if primary_wl_val is not None else "all"
                    elif param.name == "wavelength":
                        constructor_args[param.name] = primary_wl_val if primary_wl_val is not None else "primary"
                    elif param.name == "num_rays": constructor_args[param.name] = 7 if selected_analysis_name == "Ray Fan" else 24
                    elif param.name == "distribution": constructor_args[param.name] = "line_y" if selected_analysis_name == "Ray Fan" else "grid"
                    elif param.name == "num_points": constructor_args[param.name] = 50
                    elif param.name == "max_freq": constructor_args[param.name] = 100
                    elif param.name == "grid_size": constructor_args[param.name] = 10
                    elif param.name == "scale": constructor_args[param.name] = "linear"
                    elif param.name == "pupil_points": constructor_args[param.name] = 32
                
                self.logArea.append(f"Using arguments: {constructor_args}")
                analysis_instance = analysis_class(**constructor_args)
                
                # For now, view() still opens external window.
                # Store info for the new page
                new_page_data = {
                    "name": selected_analysis_name,
                    "args": constructor_args,
                    # "instance": analysis_instance # Storing instance might be heavy, consider storing data to replot
                    "result_summary": f"RMS: X.XXX (Simulated for {selected_analysis_name})" # Placeholder
                }
                self.analysis_results_pages.append(new_page_data)
                self.current_plot_page_index = len(self.analysis_results_pages) - 1
                self.update_pagination_ui()
                self.display_plot_page(self.current_plot_page_index)

                analysis_instance.view() # External plot
                
                self.logArea.append(f"{selected_analysis_name} completed and new page added.")
                print(f"Analysis Panel: Ran {selected_analysis_name}, new page {self.current_plot_page_index + 1} added.")

            except Exception as e:
                self.logArea.append(f"Error running {selected_analysis_name}: {e}")
                QMessageBox.critical(self, "Analysis Error", f"Error during {selected_analysis_name}:\n{e}")
                print(f"Analysis Panel Error: {e}")
        else:
            self.logArea.setText(f"Analysis type '{selected_analysis_name}' not yet implemented.")

    @Slot()
    def run_all_analysis_slot(self):
        self.logArea.append("Run All Analysis: Not yet implemented.")
        QMessageBox.information(self, "Not Implemented", "Run All Analysis is not yet implemented.")

    @Slot()
    def stop_analysis_slot(self):
        self.logArea.append("Stop Analysis: Not yet implemented.")
        QMessageBox.information(self, "Not Implemented", "Stop Analysis is not yet implemented.")