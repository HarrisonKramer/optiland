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

# Matplotlib Qt backend
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt # For plt.close when cleaning up

# Import Optiland analysis modules
from optiland.analysis import ( # [cite: uploaded:analysis_panel.py]
    Distortion, # [cite: uploaded:analysis_panel.py]
    EncircledEnergy, # [cite: uploaded:analysis_panel.py]
    FieldCurvature, # [cite: uploaded:analysis_panel.py]
    GridDistortion, # [cite: uploaded:analysis_panel.py]
    PupilAberration, # [cite: uploaded:analysis_panel.py]
    RayFan, # [cite: uploaded:analysis_panel.py]
    RmsSpotSizeVsField, # [cite: uploaded:analysis_panel.py]
    RmsWavefrontErrorVsField, # [cite: uploaded:analysis_panel.py]
    SpotDiagram, # [cite: uploaded:analysis_panel.py]
)
from optiland.mtf import FFTMTF, GeometricMTF # [cite: uploaded:analysis_panel.py]

try:
    from optiland.analysis.y_ybar import YYbar # [cite: uploaded:analysis_panel.py]
except ImportError as e:
    print(f"Could not import YYbar from optiland.analysis.y_ybar: {e}. Y-Ybar analysis will not be available.") # [cite: uploaded:analysis_panel.py]
    YYbar = None # [cite: uploaded:analysis_panel.py]

from .optiland_connector import OptilandConnector # [cite: uploaded:analysis_panel.py]


class AnalysisPanel(QWidget):
    ANALYSIS_MAP = { # [cite: uploaded:analysis_panel.py]
        "Spot Diagram": SpotDiagram, # [cite: uploaded:analysis_panel.py]
        "Encircled Energy": EncircledEnergy, # [cite: uploaded:analysis_panel.py]
        "Ray Fan": RayFan, # [cite: uploaded:analysis_panel.py]
        "Distortion Plot": Distortion, # [cite: uploaded:analysis_panel.py]
        "Grid Distortion": GridDistortion, # [cite: uploaded:analysis_panel.py]
        "Field Curvature": FieldCurvature, # [cite: uploaded:analysis_panel.py]
        "RMS Spot Size vs Field": RmsSpotSizeVsField, # [cite: uploaded:analysis_panel.py]
        "RMS Wavefront Error vs Field": RmsWavefrontErrorVsField, # [cite: uploaded:analysis_panel.py]
        "Pupil Aberration": PupilAberration, # [cite: uploaded:analysis_panel.py]
        "Geometric MTF": GeometricMTF, # [cite: uploaded:analysis_panel.py]
        "FFT MTF": FFTMTF, # [cite: uploaded:analysis_panel.py]
    }
    if YYbar is not None: # [cite: uploaded:analysis_panel.py]
        ANALYSIS_MAP["Y-Ybar Diagram"] = YYbar # [cite: uploaded:analysis_panel.py]
    else: # [cite: uploaded:analysis_panel.py]
        print("Note: YYbar class was not imported (from optiland.analysis.y_ybar), so 'Y-Ybar Diagram' will not be added to analysis options.") # [cite: uploaded:analysis_panel.py]

    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector # [cite: uploaded:analysis_panel.py]
        self.setWindowTitle("Analysis")

        self.setObjectName("AnalysisPanel")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.analysis_results_pages = [] # [cite: uploaded:analysis_panel.py]
        self.current_plot_page_index = -1 # [cite: uploaded:analysis_panel.py]
        self.active_mpl_canvas = None # [cite: uploaded:analysis_panel.py]
        self.active_mpl_toolbar = None # [cite: uploaded:analysis_panel.py]
        self.motion_notify_cid = None # [cite: uploaded:analysis_panel.py]

        # --- Top Bar ---
        top_bar_layout = QHBoxLayout() # [cite: uploaded:analysis_panel.py]
        top_bar_layout.addWidget(QLabel("Analysis Type:")) # [cite: uploaded:analysis_panel.py]
        self.analysisTypeCombo = QComboBox() # [cite: uploaded:analysis_panel.py]
        self.analysisTypeCombo.addItems(list(self.ANALYSIS_MAP.keys())) # [cite: uploaded:analysis_panel.py]
        self.analysisTypeCombo.setObjectName("AnalysisTypeCombo") # [cite: uploaded:analysis_panel.py]
        top_bar_layout.addWidget(self.analysisTypeCombo) # [cite: uploaded:analysis_panel.py]
        top_bar_layout.addSpacerItem( # [cite: uploaded:analysis_panel.py]
            QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self.btnRun = QPushButton("Run") # [cite: uploaded:analysis_panel.py]
        self.btnRun.setObjectName("RunAnalysisButton") # [cite: uploaded:analysis_panel.py]
        self.btnRunAll = QPushButton("Run All") # [cite: uploaded:analysis_panel.py]
        self.btnRunAll.setObjectName("RunAllAnalysisButton") # [cite: uploaded:analysis_panel.py]
        self.btnStop = QPushButton("Stop") # [cite: uploaded:analysis_panel.py]
        self.btnStop.setObjectName("StopAnalysisButton") # [cite: uploaded:analysis_panel.py]
        top_bar_layout.addWidget(self.btnRun) # [cite: uploaded:analysis_panel.py]
        top_bar_layout.addWidget(self.btnRunAll) # [cite: uploaded:analysis_panel.py]
        top_bar_layout.addWidget(self.btnStop) # [cite: uploaded:analysis_panel.py]
        main_layout.addLayout(top_bar_layout) # [cite: uploaded:analysis_panel.py]

        # --- Main Horizontal Separator ---
        main_separator_line = QFrame() # [cite: uploaded:analysis_panel.py]
        main_separator_line.setObjectName("MainSeparatorLine") # [cite: uploaded:analysis_panel.py]
        main_separator_line.setFrameShape(QFrame.Shape.HLine) # [cite: uploaded:analysis_panel.py]
        main_separator_line.setFrameShadow(QFrame.Shadow.Sunken) # [cite: uploaded:analysis_panel.py]
        main_layout.addWidget(main_separator_line) # [cite: uploaded:analysis_panel.py]

        # --- Main Content Area (Plot Display Frame + Settings Panel) ---
        main_content_layout = QHBoxLayout() # [cite: uploaded:analysis_panel.py]
        main_content_layout.setSpacing(10) # [cite: uploaded:analysis_panel.py]

        # Left/Central Area (Plot Display Frame)
        self.plot_display_frame = QFrame() # [cite: uploaded:analysis_panel.py]
        self.plot_display_frame.setObjectName("PlotDisplayFrame") # [cite: uploaded:analysis_panel.py]
        self.plot_display_frame.setFrameShape(QFrame.Shape.StyledPanel) # [cite: uploaded:analysis_panel.py]
        self.plot_display_frame.setFrameShadow(QFrame.Shadow.Plain) # [cite: uploaded:analysis_panel.py]
        plot_display_frame_layout = QVBoxLayout(self.plot_display_frame) # [cite: uploaded:analysis_panel.py]
        plot_display_frame_layout.setContentsMargins(5, 5, 5, 5) # [cite: uploaded:analysis_panel.py]
        plot_display_frame_layout.setSpacing(5) # [cite: uploaded:analysis_panel.py]

        # Plot Area Title Bar (Title + MPL Toolbar Placeholder + Settings Toggle)
        self.plot_area_title_bar_layout = QHBoxLayout() # Made instance member # [cite: uploaded:analysis_panel.py]
        self.plotTitleLabel = QLabel("No Analysis Run") # [cite: uploaded:analysis_panel.py]
        self.plotTitleLabel.setObjectName("PlotTitleLabel") # [cite: uploaded:analysis_panel.py]
        self.plot_area_title_bar_layout.addWidget(self.plotTitleLabel) # [cite: uploaded:analysis_panel.py]
        
        self.mpl_toolbar_in_titlebar_container = QWidget() # [cite: uploaded:analysis_panel.py]
        self.mpl_toolbar_in_titlebar_container.setObjectName("MPLToolbarInTitlebarContainer") # [cite: uploaded:analysis_panel.py]
        self.mpl_toolbar_in_titlebar_layout = QHBoxLayout(self.mpl_toolbar_in_titlebar_container) # [cite: uploaded:analysis_panel.py]
        self.mpl_toolbar_in_titlebar_layout.setContentsMargins(0,0,0,0) # [cite: uploaded:analysis_panel.py]
        self.mpl_toolbar_in_titlebar_layout.setSpacing(1) # Reduced spacing for toolbar items # [cite: uploaded:analysis_panel.py]
        self.plot_area_title_bar_layout.addWidget(self.mpl_toolbar_in_titlebar_container) # [cite: uploaded:analysis_panel.py]
        self.mpl_toolbar_in_titlebar_container.setVisible(False) # [cite: uploaded:analysis_panel.py]

        self.plot_area_title_bar_layout.addStretch() # [cite: uploaded:analysis_panel.py]
        self.toggleSettingsButton = QPushButton(">") # [cite: uploaded:analysis_panel.py]
        self.toggleSettingsButton.setObjectName("ToggleSettingsButton") # [cite: uploaded:analysis_panel.py]
        self.toggleSettingsButton.setFixedSize(25, 25) # [cite: uploaded:analysis_panel.py]
        self.toggleSettingsButton.setToolTip("Toggle Settings Panel") # [cite: uploaded:analysis_panel.py]
        self.plot_area_title_bar_layout.addWidget(self.toggleSettingsButton) # [cite: uploaded:analysis_panel.py]
        plot_display_frame_layout.addLayout(self.plot_area_title_bar_layout) # [cite: uploaded:analysis_panel.py]

        # Separator line below the plot title/toolbar row
        title_plot_separator_line = QFrame() # [cite: uploaded:analysis_panel.py]
        title_plot_separator_line.setObjectName("PlotTitleSeparatorLine") # [cite: uploaded:analysis_panel.py]
        title_plot_separator_line.setFrameShape(QFrame.Shape.HLine) # [cite: uploaded:analysis_panel.py]
        title_plot_separator_line.setFrameShadow(QFrame.Shadow.Sunken) # [cite: uploaded:analysis_panel.py]
        plot_display_frame_layout.addWidget(title_plot_separator_line) # [cite: uploaded:analysis_panel.py]

        # Main content (plot canvas/placeholders and page buttons)
        plot_content_and_pages_layout = QHBoxLayout() # [cite: uploaded:analysis_panel.py]
        plot_content_and_pages_layout.setContentsMargins(0,0,0,0) # [cite: uploaded:analysis_panel.py]
        plot_content_and_pages_layout.setSpacing(5) # [cite: uploaded:analysis_panel.py]

        plot_and_info_widget = QWidget() # [cite: uploaded:analysis_panel.py] # Container for plot_container and dataInfoLabel
        plot_and_info_layout = QVBoxLayout(plot_and_info_widget) # [cite: uploaded:analysis_panel.py]
        plot_and_info_layout.setContentsMargins(0,0,0,0) # [cite: uploaded:analysis_panel.py]
        plot_and_info_layout.setSpacing(5) # [cite: uploaded:analysis_panel.py]

        # This widget is the main area for the plot itself and any overlays like coordinates
        self.plot_container_widget = QWidget() # [cite: uploaded:analysis_panel.py]
        self.plot_container_widget.setObjectName("PlotContainerWidget") # [cite: uploaded:analysis_panel.py]
        self.plot_container_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # [cite: uploaded:analysis_panel.py]
        # A QVBoxLayout will be set for plot_container_widget in display_plot_page to hold the canvas/placeholders
        plot_and_info_layout.addWidget(self.plot_container_widget, 1) # [cite: uploaded:analysis_panel.py]

        # Cursor Coordinate Label - parented to plot_container_widget for overlay
        self.cursor_coord_label = QLabel("", self.plot_container_widget) # [cite: uploaded:analysis_panel.py]
        self.cursor_coord_label.setObjectName("CursorCoordLabel") # [cite: uploaded:analysis_panel.py]
        self.cursor_coord_label.setStyleSheet( # [cite: uploaded:analysis_panel.py]
            "background-color: rgba(0,0,0,0.65); color: white; padding: 2px 4px; border-radius: 3px;"
        ) 
        self.cursor_coord_label.setVisible(False) # [cite: uploaded:analysis_panel.py]
        self.cursor_coord_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents) # [cite: uploaded:analysis_panel.py]

        self.dataInfoLabel = QLabel("Data Analysis info will appear here.") # [cite: uploaded:analysis_panel.py]
        self.dataInfoLabel.setObjectName("DataInfoLabel") # [cite: uploaded:analysis_panel.py]
        plot_and_info_layout.addWidget(self.dataInfoLabel) # [cite: uploaded:analysis_panel.py]
        plot_content_and_pages_layout.addWidget(plot_and_info_widget, 1) # [cite: uploaded:analysis_panel.py]

        self.page_buttons_scroll_area = QScrollArea() # [cite: uploaded:analysis_panel.py]
        self.page_buttons_scroll_area.setObjectName("PageButtonsScrollArea") # [cite: uploaded:analysis_panel.py]
        self.page_buttons_scroll_area.setWidgetResizable(True) # [cite: uploaded:analysis_panel.py]
        self.page_buttons_scroll_area.setFixedWidth(30) # [cite: uploaded:analysis_panel.py]
        self.page_buttons_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded) # [cite: uploaded:analysis_panel.py]
        self.page_buttons_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) # [cite: uploaded:analysis_panel.py]
        page_buttons_container_widget = QWidget() # [cite: uploaded:analysis_panel.py]
        self.vertical_page_buttons_layout = QVBoxLayout(page_buttons_container_widget) # [cite: uploaded:analysis_panel.py]
        self.vertical_page_buttons_layout.setContentsMargins(2, 2, 2, 2) # [cite: uploaded:analysis_panel.py]
        self.vertical_page_buttons_layout.setSpacing(5) # [cite: uploaded:analysis_panel.py]
        self.vertical_page_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # [cite: uploaded:analysis_panel.py]
        self.page_buttons_scroll_area.setWidget(page_buttons_container_widget) # [cite: uploaded:analysis_panel.py]
        plot_content_and_pages_layout.addWidget(self.page_buttons_scroll_area) # [cite: uploaded:analysis_panel.py]

        plot_display_frame_layout.addLayout(plot_content_and_pages_layout, 1) # [cite: uploaded:analysis_panel.py]
        main_content_layout.addWidget(self.plot_display_frame, 3) # [cite: uploaded:analysis_panel.py]

        # Right Settings Area
        self.settings_area_widget = QWidget() # [cite: uploaded:analysis_panel.py]
        self.settings_area_widget.setObjectName("SettingsArea") # [cite: uploaded:analysis_panel.py]
        self.settings_area_widget.setFixedWidth(250) # [cite: uploaded:analysis_panel.py]
        settings_layout = QVBoxLayout(self.settings_area_widget) # [cite: uploaded:analysis_panel.py]
        settings_layout.setContentsMargins(5,5,5,5) # [cite: uploaded:analysis_panel.py]
        settings_title_label = QLabel("Settings here") # [cite: uploaded:analysis_panel.py]
        settings_title_label.setObjectName("SettingsTitleLabel") # [cite: uploaded:analysis_panel.py]
        settings_layout.addWidget(settings_title_label) # [cite: uploaded:analysis_panel.py]
        scroll_area_settings = QScrollArea() # [cite: uploaded:analysis_panel.py]
        scroll_area_settings.setObjectName("SettingsScrollArea") # [cite: uploaded:analysis_panel.py]
        scroll_area_settings.setWidgetResizable(True) # [cite: uploaded:analysis_panel.py]
        scroll_area_settings.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) # [cite: uploaded:analysis_panel.py]
        self.settingsContentWidget = QWidget() # [cite: uploaded:analysis_panel.py]
        self.settingsContentWidget.setObjectName("SettingsContentWidget") # [cite: uploaded:analysis_panel.py]
        settings_form_layout = QFormLayout(self.settingsContentWidget) # [cite: uploaded:analysis_panel.py]
        settings_form_layout.setContentsMargins(10,10,10,10) # [cite: uploaded:analysis_panel.py]
        settings_form_layout.setSpacing(8) # [cite: uploaded:analysis_panel.py]
        settings_form_layout.addRow("Setting Option A:", QLabel("Value")) # [cite: uploaded:analysis_panel.py]
        settings_form_layout.addRow("Setting Option B:", QLabel("Another Value")) # [cite: uploaded:analysis_panel.py]
        settings_form_layout.addRow(QLabel("More settings can go here,\nand this area will scroll if needed.")) # [cite: uploaded:analysis_panel.py]
        settings_form_layout.addRow("Parameter X:", QLabel("True")) # [cite: uploaded:analysis_panel.py]
        settings_form_layout.addRow("Parameter Y:", QLabel("123.45")) # [cite: uploaded:analysis_panel.py]
        settings_form_layout.addRow("Wavelength:", QLabel("0.550 µm")) # [cite: uploaded:analysis_panel.py]
        settings_form_layout.addRow("Number of Rays:", QLabel("1000")) # [cite: uploaded:analysis_panel.py]
        settings_form_layout.addRow("Show Airy Disk:", QLabel("Yes")) # [cite: uploaded:analysis_panel.py]
        scroll_area_settings.setWidget(self.settingsContentWidget) # [cite: uploaded:analysis_panel.py]
        settings_layout.addWidget(scroll_area_settings, 1) # [cite: uploaded:analysis_panel.py]
        main_content_layout.addWidget(self.settings_area_widget, 1) # [cite: uploaded:analysis_panel.py]
        main_layout.addLayout(main_content_layout, 1) # [cite: uploaded:analysis_panel.py]

        # --- Bottom Log Area ---
        self.logArea = QTextEdit() # [cite: uploaded:analysis_panel.py]
        self.logArea.setObjectName("LogArea") # [cite: uploaded:analysis_panel.py]
        self.logArea.setReadOnly(True) # [cite: uploaded:analysis_panel.py]
        self.logArea.setPlaceholderText("Select an analysis and click 'Run'.") # [cite: uploaded:analysis_panel.py]
        self.logArea.setFixedHeight(60) # [cite: uploaded:analysis_panel.py]
        main_layout.addWidget(self.logArea) # [cite: uploaded:analysis_panel.py]

        self.btnRun.clicked.connect(self.run_analysis_slot) # [cite: uploaded:analysis_panel.py]
        self.btnRunAll.clicked.connect(self.run_all_analysis_slot) # [cite: uploaded:analysis_panel.py]
        self.btnStop.clicked.connect(self.stop_analysis_slot) # [cite: uploaded:analysis_panel.py]
        self.analysisTypeCombo.currentTextChanged.connect(self.update_ui_for_analysis_type) # [cite: uploaded:analysis_panel.py]
        self.toggleSettingsButton.clicked.connect(self.toggle_settings_panel_slot) # [cite: uploaded:analysis_panel.py]

        self.update_ui_for_analysis_type(self.analysisTypeCombo.currentText()) # [cite: uploaded:analysis_panel.py]
        self.update_pagination_ui() # [cite: uploaded:analysis_panel.py]
        self.display_plot_page(self.current_plot_page_index) # [cite: uploaded:analysis_panel.py]
        self.settings_area_widget.setVisible(False) # [cite: uploaded:analysis_panel.py]
        self.toggleSettingsButton.setText(">") # [cite: uploaded:analysis_panel.py]

    def _clear_layout(self, layout_to_clear): # [cite: uploaded:analysis_panel.py]
        if layout_to_clear is not None: # [cite: uploaded:analysis_panel.py]
            while layout_to_clear.count(): # [cite: uploaded:analysis_panel.py]
                item = layout_to_clear.takeAt(0) # [cite: uploaded:analysis_panel.py]
                widget = item.widget() # [cite: uploaded:analysis_panel.py]
                if widget is not None: # [cite: uploaded:analysis_panel.py]
                    if isinstance(widget, FigureCanvas): # [cite: uploaded:analysis_panel.py]
                        if hasattr(widget, '_motion_notify_cid') and widget._motion_notify_cid is not None: # [cite: uploaded:analysis_panel.py]
                            try: # [cite: uploaded:analysis_panel.py]
                                widget.mpl_disconnect(widget._motion_notify_cid) # [cite: uploaded:analysis_panel.py]
                            except TypeError: # [cite: uploaded:analysis_panel.py]
                                pass # [cite: uploaded:analysis_panel.py]
                            widget._motion_notify_cid = None # [cite: uploaded:analysis_panel.py]
                        plt.close(widget.figure) # [cite: uploaded:analysis_panel.py]
                    widget.deleteLater() # [cite: uploaded:analysis_panel.py]
                else: # [cite: uploaded:analysis_panel.py]
                    sub_layout = item.layout() # [cite: uploaded:analysis_panel.py]
                    if sub_layout is not None: # [cite: uploaded:analysis_panel.py]
                        self._clear_layout(sub_layout) # [cite: uploaded:analysis_panel.py]

    def update_pagination_ui(self): # [cite: uploaded:analysis_panel.py]
        self._clear_layout(self.vertical_page_buttons_layout) # [cite: uploaded:analysis_panel.py]
        for i, page_data in enumerate(self.analysis_results_pages): # [cite: uploaded:analysis_panel.py]
            btn_page = QPushButton(str(i + 1)) # [cite: uploaded:analysis_panel.py]
            btn_page.setObjectName(f"PageButton_{i+1}") # [cite: uploaded:analysis_panel.py]
            btn_page.setCheckable(True) # [cite: uploaded:analysis_panel.py]
            btn_page.setChecked(i == self.current_plot_page_index) # [cite: uploaded:analysis_panel.py]
            btn_page.clicked.connect(lambda checked=False, index=i: self.switch_plot_page(index)) # [cite: uploaded:analysis_panel.py]
            self.vertical_page_buttons_layout.addWidget(btn_page) # [cite: uploaded:analysis_panel.py]
        self.vertical_page_buttons_layout.addStretch() # [cite: uploaded:analysis_panel.py]

    def switch_plot_page(self, page_index): # [cite: uploaded:analysis_panel.py]
        if 0 <= page_index < len(self.analysis_results_pages): # [cite: uploaded:analysis_panel.py]
            if self.current_plot_page_index == page_index and self.active_mpl_canvas is not None: # [cite: uploaded:analysis_panel.py]
                return # [cite: uploaded:analysis_panel.py]

            self.current_plot_page_index = page_index # [cite: uploaded:analysis_panel.py]
            self.update_pagination_ui() # [cite: uploaded:analysis_panel.py]
            self.display_plot_page(page_index) # [cite: uploaded:analysis_panel.py]
            self.logArea.append(f"Switched to page {page_index + 1}: {self.analysis_results_pages[page_index].get('name', 'Analysis')}") # [cite: uploaded:analysis_panel.py]
        else: # [cite: uploaded:analysis_panel.py]
            self.current_plot_page_index = -1 # [cite: uploaded:analysis_panel.py]
            self.update_pagination_ui() # [cite: uploaded:analysis_panel.py]
            self.display_plot_page(-1) # [cite: uploaded:analysis_panel.py]

    def on_mouse_move_on_plot(self, event): # [cite: uploaded:analysis_panel.py]
        if event.inaxes and self.active_mpl_canvas: # [cite: uploaded:analysis_panel.py]
            x_coord = f"{event.xdata:.2f}" if event.xdata is not None else "---" # [cite: uploaded:analysis_panel.py]
            y_coord = f"{event.ydata:.2f}" if event.ydata is not None else "---" # [cite: uploaded:analysis_panel.py]
            self.cursor_coord_label.setText(f"(x, y) = ({x_coord}, {y_coord})") # [cite: uploaded:analysis_panel.py]
            self.cursor_coord_label.adjustSize() # [cite: uploaded:analysis_panel.py]
            self.cursor_coord_label.move(5, 5)  # Position in top-left of plot_container_widget # [cite: uploaded:analysis_panel.py]
            self.cursor_coord_label.setVisible(True) # [cite: uploaded:analysis_panel.py]
            self.cursor_coord_label.raise_() # [cite: uploaded:analysis_panel.py]
        elif self.active_mpl_canvas: # [cite: uploaded:analysis_panel.py]
            self.cursor_coord_label.setVisible(False) # [cite: uploaded:analysis_panel.py]

    def display_plot_page(self, page_index): # [cite: uploaded:analysis_panel.py]
        # Clear previous toolbar from its dedicated container (now inside title bar)
        if self.active_mpl_toolbar: # [cite: uploaded:analysis_panel.py]
            self.mpl_toolbar_in_titlebar_layout.removeWidget(self.active_mpl_toolbar) # [cite: uploaded:analysis_panel.py]
            self.active_mpl_toolbar.deleteLater() # [cite: uploaded:analysis_panel.py]
            self.active_mpl_toolbar = None # [cite: uploaded:analysis_panel.py]
        self.mpl_toolbar_in_titlebar_container.setVisible(False) # [cite: uploaded:analysis_panel.py]

        # Disconnect previous motion_notify_event if canvas existed
        if self.active_mpl_canvas and self.motion_notify_cid: # [cite: uploaded:analysis_panel.py]
            try: # [cite: uploaded:analysis_panel.py]
                self.active_mpl_canvas.mpl_disconnect(self.motion_notify_cid) # [cite: uploaded:analysis_panel.py]
            except TypeError: pass # [cite: uploaded:analysis_panel.py]
            self.motion_notify_cid = None # [cite: uploaded:analysis_panel.py]
        self.cursor_coord_label.setVisible(False) # [cite: uploaded:analysis_panel.py]

        # Clear previous plot content from plot_container_widget's layout
        plot_content_area_layout = self.plot_container_widget.layout() # [cite: uploaded:analysis_panel.py]
        if plot_content_area_layout is not None: # [cite: uploaded:analysis_panel.py]
            self._clear_layout(plot_content_area_layout) # [cite: uploaded:analysis_panel.py]
        else:  # [cite: uploaded:analysis_panel.py]
            plot_content_area_layout = QVBoxLayout(self.plot_container_widget) # [cite: uploaded:analysis_panel.py]
            plot_content_area_layout.setContentsMargins(0,0,0,0) # [cite: uploaded:analysis_panel.py]
            self.plot_container_widget.setLayout(plot_content_area_layout) # [cite: uploaded:analysis_panel.py]
        
        self.active_mpl_canvas = None # [cite: uploaded:analysis_panel.py]

        if 0 <= page_index < len(self.analysis_results_pages): # [cite: uploaded:analysis_panel.py]
            page_data = self.analysis_results_pages[page_index] # [cite: uploaded:analysis_panel.py]
            analysis_name = page_data.get("name", "Analysis") # [cite: uploaded:analysis_panel.py]
            self.plotTitleLabel.setText(analysis_name) # [cite: uploaded:analysis_panel.py]
            self.dataInfoLabel.setText(page_data.get("result_summary", f"Displaying results for {analysis_name}")) # [cite: uploaded:analysis_panel.py]

            plot_content = page_data.get("plot_content") # [cite: uploaded:analysis_panel.py]
            if isinstance(plot_content, FigureCanvas): # [cite: uploaded:analysis_panel.py]
                self.active_mpl_canvas = plot_content # [cite: uploaded:analysis_panel.py]
                
                self.active_mpl_toolbar = NavigationToolbar(self.active_mpl_canvas, self.mpl_toolbar_in_titlebar_container) # [cite: uploaded:analysis_panel.py]
                self.active_mpl_toolbar.setObjectName("AnalysisPlotToolbarTitle") # [cite: uploaded:analysis_panel.py]
                def do_nothing_with_message(s): pass # pylint: disable=unused-argument # [cite: uploaded:analysis_panel.py]
                self.active_mpl_toolbar.set_message = do_nothing_with_message # [cite: uploaded:analysis_panel.py]
                self.mpl_toolbar_in_titlebar_layout.addWidget(self.active_mpl_toolbar) # [cite: uploaded:analysis_panel.py]
                self.mpl_toolbar_in_titlebar_container.setVisible(True) # [cite: uploaded:analysis_panel.py]
                
                plot_content_area_layout.addWidget(self.active_mpl_canvas) # Add canvas to its dedicated content widget # [cite: uploaded:analysis_panel.py]
                self.motion_notify_cid = self.active_mpl_canvas.mpl_connect('motion_notify_event', self.on_mouse_move_on_plot) # [cite: uploaded:analysis_panel.py]
                self.active_mpl_canvas._motion_notify_cid = self.motion_notify_cid # [cite: uploaded:analysis_panel.py]

            elif plot_content == "spot_diagram_placeholders": # [cite: uploaded:analysis_panel.py]
                grid_widget = QWidget() # [cite: uploaded:analysis_panel.py]
                grid_layout = QGridLayout(grid_widget) # [cite: uploaded:analysis_panel.py]
                ph1 = QLabel(f"Page {page_index+1}\nSpot Diagram\nField 1"); ph1.setAlignment(Qt.AlignCenter); ph1.setFrameShape(QFrame.StyledPanel); ph1.setMinimumSize(150,150) # [cite: uploaded:analysis_panel.py]
                ph2 = QLabel(f"Page {page_index+1}\nSpot Diagram\nField 2"); ph2.setAlignment(Qt.AlignCenter); ph2.setFrameShape(QFrame.StyledPanel); ph2.setMinimumSize(150,150) # [cite: uploaded:analysis_panel.py]
                ph3 = QLabel(f"Page {page_index+1}\nSpot Diagram\nField 3"); ph3.setAlignment(Qt.AlignCenter); ph3.setFrameShape(QFrame.StyledPanel); ph3.setMinimumSize(150,150) # [cite: uploaded:analysis_panel.py]
                grid_layout.addWidget(ph1,0,0); grid_layout.addWidget(ph2,0,1); grid_layout.addWidget(ph3,0,2) # [cite: uploaded:analysis_panel.py]
                plot_content_area_layout.addWidget(grid_widget) # [cite: uploaded:analysis_panel.py]
            else: # [cite: uploaded:analysis_panel.py]
                placeholder_label = QLabel(f"Content for {analysis_name} (Page {page_index + 1})\n(External window or not yet embedded)") # [cite: uploaded:analysis_panel.py]
                placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # [cite: uploaded:analysis_panel.py]
                placeholder_label.setFrameShape(QFrame.Shape.StyledPanel) # [cite: uploaded:analysis_panel.py]
                plot_content_area_layout.addWidget(placeholder_label) # [cite: uploaded:analysis_panel.py]
        else: # [cite: uploaded:analysis_panel.py]
            self.plotTitleLabel.setText("No Analysis Selected") # [cite: uploaded:analysis_panel.py]
            self.dataInfoLabel.setText("Run an analysis to see results.") # [cite: uploaded:analysis_panel.py]
            grid_widget = QWidget() # [cite: uploaded:analysis_panel.py]
            grid_layout = QGridLayout(grid_widget) # [cite: uploaded:analysis_panel.py]
            ph1 = QLabel("Plot Area 1"); ph1.setAlignment(Qt.AlignCenter); ph1.setFrameShape(QFrame.StyledPanel); ph1.setMinimumSize(150,150) # [cite: uploaded:analysis_panel.py]
            ph2 = QLabel("Plot Area 2"); ph2.setAlignment(Qt.AlignCenter); ph2.setFrameShape(QFrame.StyledPanel); ph2.setMinimumSize(150,150) # [cite: uploaded:analysis_panel.py]
            ph3 = QLabel("Plot Area 3"); ph3.setAlignment(Qt.AlignCenter); ph3.setFrameShape(QFrame.StyledPanel); ph3.setMinimumSize(150,150) # [cite: uploaded:analysis_panel.py]
            grid_layout.addWidget(ph1,0,0); grid_layout.addWidget(ph2,0,1); grid_layout.addWidget(ph3,0,2) # [cite: uploaded:analysis_panel.py]
            plot_content_area_layout.addWidget(grid_widget) # [cite: uploaded:analysis_panel.py]


    @Slot(str)
    def update_ui_for_analysis_type(self, analysis_name): # [cite: uploaded:analysis_panel.py]
        if self.current_plot_page_index == -1 or not self.analysis_results_pages : # [cite: uploaded:analysis_panel.py]
             self.plotTitleLabel.setText(analysis_name) # [cite: uploaded:analysis_panel.py]

    @Slot()
    def toggle_settings_panel_slot(self): # [cite: uploaded:analysis_panel.py]
        if self.settings_area_widget.isVisible(): # [cite: uploaded:analysis_panel.py]
            self.settings_area_widget.setVisible(False) # [cite: uploaded:analysis_panel.py]
            self.toggleSettingsButton.setText(">") # [cite: uploaded:analysis_panel.py]
        else: # [cite: uploaded:analysis_panel.py]
            self.settings_area_widget.setVisible(True) # [cite: uploaded:analysis_panel.py]
            self.toggleSettingsButton.setText("<") # [cite: uploaded:analysis_panel.py]

    @Slot()
    def run_analysis_slot(self): # [cite: uploaded:analysis_panel.py]
        selected_analysis_name = self.analysisTypeCombo.currentText() # [cite: uploaded:analysis_panel.py]
        analysis_class = self.ANALYSIS_MAP.get(selected_analysis_name) # [cite: uploaded:analysis_panel.py]
        optic = self.connector.get_optic() # [cite: uploaded:analysis_panel.py]

        if not optic or optic.surface_group.num_surfaces <= 2: # [cite: uploaded:analysis_panel.py]
            QMessageBox.warning(self, "Analysis Error", "Cannot run analysis on an empty or minimal system.") # [cite: uploaded:analysis_panel.py]
            return # [cite: uploaded:analysis_panel.py]
        if optic.wavelengths.num_wavelengths == 0: # [cite: uploaded:analysis_panel.py]
            QMessageBox.warning(self, "Analysis Error", "Optic has no wavelengths defined.") # [cite: uploaded:analysis_panel.py]
            return # [cite: uploaded:analysis_panel.py]

        if analysis_class: # [cite: uploaded:analysis_panel.py]
            self.logArea.setText(f"Running {selected_analysis_name}...") # [cite: uploaded:analysis_panel.py]
            try:
                primary_wl_val = self.connector._get_safe_primary_wavelength_value() # [cite: uploaded:analysis_panel.py]
                constructor_args = {} # [cite: uploaded:analysis_panel.py]
                for param in inspect.signature(analysis_class.__init__).parameters.values(): # type: ignore # [cite: uploaded:analysis_panel.py]
                    if param.name == "self": continue # [cite: uploaded:analysis_panel.py]
                    if param.name in ["optic", "optical_system"]: constructor_args[param.name] = optic # [cite: uploaded:analysis_panel.py]
                    elif param.name == "fields": constructor_args[param.name] = "all" # [cite: uploaded:analysis_panel.py]
                    elif param.name == "wavelengths": constructor_args[param.name] = [primary_wl_val] if primary_wl_val is not None else "all" # [cite: uploaded:analysis_panel.py]
                    elif param.name == "wavelength": constructor_args[param.name] = primary_wl_val if primary_wl_val is not None else "primary" # [cite: uploaded:analysis_panel.py]
                    elif param.name == "num_rays": constructor_args[param.name] = 7 if selected_analysis_name == "Ray Fan" else 24 # [cite: uploaded:analysis_panel.py]
                    elif param.name == "distribution": constructor_args[param.name] = "line_y" if selected_analysis_name == "Ray Fan" else "grid" # [cite: uploaded:analysis_panel.py]
                    elif param.name == "num_points": constructor_args[param.name] = 50 # [cite: uploaded:analysis_panel.py]
                    elif param.name == "max_freq": constructor_args[param.name] = 100 # [cite: uploaded:analysis_panel.py]
                    elif param.name == "grid_size": constructor_args[param.name] = 10 # [cite: uploaded:analysis_panel.py]
                    elif param.name == "scale": constructor_args[param.name] = "linear" # [cite: uploaded:analysis_panel.py]
                    elif param.name == "pupil_points": constructor_args[param.name] = 32 # [cite: uploaded:analysis_panel.py]

                analysis_instance = analysis_class(**constructor_args) # [cite: uploaded:analysis_panel.py]
                new_page_data = { "name": selected_analysis_name, "args": constructor_args } # [cite: uploaded:analysis_panel.py]

                if selected_analysis_name == "Y-Ybar Diagram" and YYbar is not None: # [cite: uploaded:analysis_panel.py]
                    fig = Figure(figsize=(5, 4), dpi=100) # [cite: uploaded:analysis_panel.py]
                    canvas = FigureCanvas(fig) # [cite: uploaded:analysis_panel.py]
                    analysis_instance.view(fig_to_plot_on=fig) # [cite: uploaded:analysis_panel.py]
                    new_page_data["plot_content"] = canvas # [cite: uploaded:analysis_panel.py]
                    new_page_data["result_summary"] = "Y-Ybar Diagram plotted in GUI." # [cite: uploaded:analysis_panel.py]
                    self.logArea.append(f"{selected_analysis_name} plotted in GUI.") # [cite: uploaded:analysis_panel.py]
                elif selected_analysis_name == "Spot Diagram": # [cite: uploaded:analysis_panel.py]
                    analysis_instance.view() # [cite: uploaded:analysis_panel.py]
                    new_page_data["plot_content"] = "spot_diagram_placeholders" # [cite: uploaded:analysis_panel.py]
                    new_page_data["result_summary"] = "Spot Diagram (external window)." # [cite: uploaded:analysis_panel.py]
                    self.logArea.append(f"{selected_analysis_name} shown in external window.") # [cite: uploaded:analysis_panel.py]
                else: # [cite: uploaded:analysis_panel.py]
                    analysis_instance.view() # [cite: uploaded:analysis_panel.py]
                    new_page_data["plot_content"] = None # [cite: uploaded:analysis_panel.py]
                    new_page_data["result_summary"] = f"{selected_analysis_name} (external window)." # [cite: uploaded:analysis_panel.py]
                    self.logArea.append(f"{selected_analysis_name} shown in external window.") # [cite: uploaded:analysis_panel.py]

                self.analysis_results_pages.append(new_page_data) # [cite: uploaded:analysis_panel.py]
                new_page_index = len(self.analysis_results_pages) - 1 # [cite: uploaded:analysis_panel.py]
                self.switch_plot_page(new_page_index) # [cite: uploaded:analysis_panel.py]
                
                print(f"Analysis Panel: Ran {selected_analysis_name}, page {new_page_index + 1} added/displayed.") # [cite: uploaded:analysis_panel.py]

            except Exception as e: # [cite: uploaded:analysis_panel.py]
                self.logArea.append(f"Error running {selected_analysis_name}: {e}") # [cite: uploaded:analysis_panel.py]
                QMessageBox.critical(self, "Analysis Error", f"Error during {selected_analysis_name}:\n{e}") # [cite: uploaded:analysis_panel.py]
                print(f"Analysis Panel Error: {e}") # [cite: uploaded:analysis_panel.py]
        else: # [cite: uploaded:analysis_panel.py]
            self.logArea.setText(f"Analysis type '{selected_analysis_name}' not yet implemented.") # [cite: uploaded:analysis_panel.py]

    @Slot()
    def run_all_analysis_slot(self): # [cite: uploaded:analysis_panel.py]
        self.logArea.append("Run All Analysis: Not yet implemented.") # [cite: uploaded:analysis_panel.py]
        QMessageBox.information(self, "Not Implemented", "Run All Analysis is not yet implemented.") # [cite: uploaded:analysis_panel.py]

    @Slot()
    def stop_analysis_slot(self): # [cite: uploaded:analysis_panel.py]
        self.logArea.append("Stop Analysis: Not yet implemented.") # [cite: uploaded:analysis_panel.py]
        QMessageBox.information(self, "Not Implemented", "Stop Analysis is not yet implemented.") # [cite: uploaded:analysis_panel.py]

