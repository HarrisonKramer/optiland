"""
Provides the main viewing panel for optical systems.

This module defines the `ViewerPanel`, which contains tabs for 2D and 3D
visualizations of the optical system. It includes `MatplotlibViewer` for 2D
plots and `VTKViewer` for 3D rendering.

@author: Manuel Fragata Mendes, 2025
"""

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False


from optiland.visualization.system.rays import Rays2D, Rays3D
from optiland.visualization.system.system import (
    OpticalSystem as OptilandOpticalSystemPlotter,
)

from . import gui_plot_utils
from .analysis_panel import CustomMatplotlibToolbar
from .optiland_connector import OptilandConnector


class ViewerPanel(QWidget):
    """
    A widget that contains multiple viewers for the optical system.

    This panel uses a QTabWidget to host different types of viewers, such as
    a 2D plot and a 3D rendering of the system.

    Attributes:
        connector (OptilandConnector): The connector to the main application logic.
        tabWidget (QTabWidget): The widget hosting the different viewer tabs.
        viewer2D (MatplotlibViewer): The 2D viewer widget.
        viewer3D (VTKViewer or QLabel): The 3D viewer widget, or a label if VTK
                                        is unavailable.
    """

    def __init__(self, connector: OptilandConnector, parent=None):
        """
        Initializes the ViewerPanel.

        Args:
            connector (OptilandConnector): The connector to the main application logic.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Viewer")

        self.layout = QVBoxLayout(self)
        self.tabWidget = QTabWidget()
        self.layout.addWidget(self.tabWidget)

        self.viewer2D = MatplotlibViewer(self.connector, self)
        self.tabWidget.addTab(self.viewer2D, "2D View")

        self.viewer3D = (
            VTKViewer(self.connector, self)
            if VTK_AVAILABLE
            else QLabel("VTK not available or not installed.")
        )
        self.tabWidget.addTab(self.viewer3D, "3D View")

        self.connector.opticLoaded.connect(self.update_viewers)
        self.connector.opticChanged.connect(self.update_viewers)

    def update_theme(self, theme="dark"):
        """
        Updates the theme for all viewers in the panel.

        Args:
            theme (str, optional): The theme name ('dark' or 'light').
                                   Defaults to "dark".
        """
        self.viewer2D.update_theme(theme)
        if VTK_AVAILABLE:
            self.viewer3D.update_theme(theme)

    @Slot()
    def update_viewers(self):
        """Updates the content of all viewers."""
        self.viewer2D.plot_optic()
        if VTK_AVAILABLE:
            self.viewer3D.render_optic()


class MatplotlibViewer(QWidget):
    """
    A widget for displaying a 2D plot of the optical system using Matplotlib.

    This viewer includes a Matplotlib canvas, a custom toolbar, and a settings
    panel for controlling the plot, such as the number of rays to trace.

    Attributes:
        figure (Figure): The Matplotlib figure object.
        canvas (FigureCanvas): The canvas widget that displays the figure.
        ax (Axes): The Matplotlib axes object for plotting.
        toolbar (CustomMatplotlibToolbar): The toolbar for plot navigation.
        settings_area (QWidget): The panel for viewer-specific settings.
    """

    def __init__(self, connector: OptilandConnector, parent=None):
        """
        Initializes the MatplotlibViewer.

        Args:
            connector (OptilandConnector): The connector to the main application logic.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.connector = connector
        self.current_theme = "dark"

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        viewer_widget = QWidget()
        self.layout = QVBoxLayout(viewer_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        main_layout.addWidget(viewer_widget, 1)

        self.toolbar_container = QWidget()
        self.toolbar_container.setObjectName("ViewerToolbarContainer")
        toolbar_layout = QHBoxLayout(self.toolbar_container)
        toolbar_layout.setContentsMargins(5, 0, 5, 0)
        self.layout.addWidget(self.toolbar_container)

        plot_container = QWidget()
        plot_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(plot_container, 1)

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)

        self.toolbar = CustomMatplotlibToolbar(self.canvas, self.toolbar_container)
        toolbar_layout.addWidget(self.toolbar)
        toolbar_layout.addStretch()

        for action in self.toolbar.actions():
            if action.toolTip() == "Reset original view":
                # Disconnect the default trigger
                action.triggered.disconnect()
                # Connect our full plot refresh method
                action.triggered.connect(self.plot_optic)
                break

        self.settings_area = QWidget()
        self.settings_area.setObjectName("ViewerSettingsArea")
        self.settings_area.setFixedWidth(200)
        self.settings_area.setVisible(False)
        settings_layout = QVBoxLayout(self.settings_area)
        self.settings_form_layout = QFormLayout()

        self.num_rays_spinbox = QSpinBox()
        self.num_rays_spinbox.setRange(1, 100)
        self.num_rays_spinbox.setValue(3)
        self.settings_form_layout.addRow("Num Rays:", self.num_rays_spinbox)

        self.dist_combo = QComboBox()
        self.dist_combo.addItems(["line_y", "line_x", "hexapolar", "random"])
        self.settings_form_layout.addRow("Distribution:", self.dist_combo)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.plot_optic)

        settings_layout.addLayout(self.settings_form_layout)
        settings_layout.addStretch()
        settings_layout.addWidget(apply_button)
        main_layout.addWidget(self.settings_area)

        self.settings_toggle_btn = QToolButton()
        self.settings_toggle_btn.setToolTip("Toggle Viewer Settings")
        self.settings_toggle_btn.setCheckable(True)
        self.settings_toggle_btn.toggled.connect(self.settings_area.setVisible)
        self.toolbar.addWidget(self.settings_toggle_btn)

        self.cursor_coord_label = QLabel("", self.canvas)
        self.cursor_coord_label.setObjectName("CursorCoordLabel")
        self.cursor_coord_label.setStyleSheet(
            "background-color:rgba(0,0,0,0.65);color:white;padding:2px 4px;"
            "border-radius:3px;"
        )
        self.cursor_coord_label.setVisible(False)
        self.cursor_coord_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move_on_plot)
        self.canvas.mpl_connect("scroll_event", self.on_scroll_zoom)

        self.plot_optic()
        self.update_theme()

    def on_mouse_move_on_plot(self, event):
        """
        Displays the cursor's coordinates on the plot.

        Args:
            event: The Matplotlib motion notify event.
        """
        if event.inaxes:
            x_coord = f"{event.xdata:.3f}"
            y_coord = f"{event.ydata:.3f}"
            self.cursor_coord_label.setText(f"(Z, Y) = ({x_coord}, {y_coord})")
            self.cursor_coord_label.adjustSize()
            self.cursor_coord_label.move(5, 5)
            self.cursor_coord_label.setVisible(True)
            self.cursor_coord_label.raise_()
        else:
            self.cursor_coord_label.setVisible(False)

    def on_scroll_zoom(self, event):
        """
        Implements zoom functionality using the mouse scroll wheel.

        Args:
            event: The Matplotlib scroll event.
        """
        if not event.inaxes:
            return

        ax = event.inaxes
        scale_factor = 1.1 if event.step < 0 else 1 / 1.1

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - rel_x), xdata + new_width * rel_x])
        ax.set_ylim([ydata - new_height * (1 - rel_y), ydata + new_height * rel_y])
        ax.figure.canvas.draw_idle()

    def update_theme(self, theme="dark"):
        """
        Updates the theme of the Matplotlib plot.

        Args:
            theme (str, optional): The theme name ('dark' or 'light').
                                   Defaults to "dark".
        """
        if self.current_theme != theme:
            self.current_theme = theme
            gui_plot_utils.apply_gui_matplotlib_styles(theme=self.current_theme)
            self.plot_optic()
        self.settings_toggle_btn.setIcon(QIcon(f":/icons/{theme}/settings.svg"))

    def plot_optic(self):
        """
        Clears the current plot and redraws the optical system.

        This method retrieves the current optical system from the connector and
        uses Optiland's plotting utilities to generate a 2D layout.
        """
        gui_plot_utils.apply_gui_matplotlib_styles(theme=self.current_theme)

        self.ax.clear()
        face_color = matplotlib.rcParams["figure.facecolor"]
        self.figure.set_facecolor(face_color)
        self.ax.set_facecolor(face_color)

        optic = self.connector.get_optic()
        num_rays = self.num_rays_spinbox.value()
        distribution = self.dist_combo.currentText()
        if optic and optic.surface_group.num_surfaces > 0:
            try:
                rays2d_plotter = Rays2D(optic)
                system_plotter = OptilandOpticalSystemPlotter(
                    optic, rays2d_plotter, projection="2d"
                )
                rays2d_plotter.plot(
                    self.ax,
                    fields="all",
                    wavelengths="primary",
                    num_rays=num_rays,
                    distribution=distribution,
                )
                system_plotter.plot(self.ax)
                self.ax.set_title(
                    f"System: {optic.name} (2D)",
                    color=matplotlib.rcParams["text.color"],
                )
                self.ax.set_xlabel("Z-axis (mm)")
                self.ax.set_ylabel("Y-axis (mm)")
                self.ax.axis("equal")
                self.ax.grid(True, linestyle="--", alpha=0.7)
            except Exception as e:
                self.ax.text(
                    0.5,
                    0.5,
                    f"Error plotting Optiland 2D view:\n{e}",
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                    color="red",
                )
                print(f"MatplotlibViewer Error: {e}")
        else:
            self.ax.text(
                0.5,
                0.5,
                "2D Viewer (Matplotlib)\nNo Optic data or empty system.",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )

        self.canvas.draw()


class VTKViewer(QWidget):
    """
    A widget for displaying a 3D rendering of the optical system using VTK.

    This viewer embeds a QVTKRenderWindowInteractor to provide an interactive
    3D view of the optical system and traced rays.

    Attributes:
        vtkWidget (QVTKRenderWindowInteractor): The VTK render window interactor widget.
        renderer (vtkRenderer): The VTK renderer for the scene.
        iren (vtkRenderWindowInteractor): The interactor for camera manipulation.
    """

    def __init__(self, connector: OptilandConnector, parent=None):
        """
        Initializes the VTKViewer.

        Args:
            connector (OptilandConnector): The connector to the main application logic.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)

        self.connector = connector
        if not VTK_AVAILABLE:
            self.layout = QVBoxLayout(self)
            self.layout.addWidget(QLabel("VTK is not available."))
            return

        self.layout = QVBoxLayout(self)
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtkWidget)

        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.setup_default_camera()
        self.iren.Initialize()

    def setup_default_camera(self):
        """Sets up the default camera position and orientation for the 3D view."""
        self.renderer.SetBackground(0.1, 0.2, 0.4)
        camera = self.renderer.GetActiveCamera()
        if camera:
            camera.SetPosition(0.2, 0, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 1, 0)
            self.renderer.ResetCamera()
            camera.Elevation(0)
            camera.Azimuth(150)

    def update_theme(self, theme="dark"):
        """
        Updates the background color of the VTK renderer based on the theme.

        Args:
            theme (str, optional): The theme name ('dark' or 'light').
                                   Defaults to "dark".
        """
        if theme == "dark":
            self.renderer.SetBackground(0.1, 0.2, 0.4)
        else:
            self.renderer.SetBackground(0.8, 0.8, 0.8)
        self.vtkWidget.GetRenderWindow().Render()

    def render_optic(self):
        """
        Clears the current scene and re-renders the optical system in 3D.

        This method retrieves the current optical system and uses Optiland's
        VTK plotting utilities to generate the 3D visualization.
        """
        if not VTK_AVAILABLE:
            return

        self.renderer.RemoveAllViewProps()
        optic = self.connector.get_optic()

        if optic and optic.surface_group.num_surfaces > 0:
            try:
                rays3d_plotter = Rays3D(optic)
                system_plotter = OptilandOpticalSystemPlotter(
                    optic, rays3d_plotter, projection="3d"
                )

                rays3d_plotter.plot(
                    self.renderer,
                    fields="all",
                    wavelengths="primary",
                    num_rays=24,
                    distribution="ring",
                )

                system_plotter.plot(self.renderer)

                if not self.renderer.GetActiveCamera():
                    self.setup_default_camera()
                else:
                    self.renderer.ResetCameraClippingRange()
                    self.renderer.ResetCamera()

            except Exception as e:
                print(f"VTKViewer Error: {e}")
                textActor = vtk.vtkTextActor()
                textActor.SetInput(f"Error rendering 3D view:\n{e}")
                textActor.GetTextProperty().SetColor(1, 0, 0)
                self.renderer.AddActor2D(textActor)
        else:
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetRadius(0.1)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphereSource.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self.renderer.AddActor(actor)
            if not self.renderer.GetActiveCamera():
                self.setup_default_camera()
            else:
                self.renderer.ResetCamera()

        self.vtkWidget.GetRenderWindow().Render()
