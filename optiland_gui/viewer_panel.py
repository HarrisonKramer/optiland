# optiland_gui/viewer_panel.py
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget, QSizePolicy, QHBoxLayout
from . import gui_plot_utils
from .analysis_panel import CustomMatplotlibToolbar

try:
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

# Import Optiland's visualization classes
from optiland.visualization.rays import Rays2D, Rays3D  # For direct use with system
from optiland.visualization.system import OpticalSystem as OptilandOpticalSystemPlotter

from .optiland_connector import OptilandConnector


class ViewerPanel(QWidget):
    def __init__(self, connector: OptilandConnector, parent=None):
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

        self.connector.opticLoaded.connect(
            self.update_viewers
        )  # Full update on new load
        self.connector.opticChanged.connect(self.update_viewers)  # Update on any change

    def update_theme(self, theme="dark"):
        """Propagates theme changes to child viewers."""
        self.viewer2D.update_theme(theme)

    @Slot()
    def update_viewers(self):
        print("ViewerPanel: Updating viewers due to optic change.")
        self.viewer2D.plot_optic()
        if VTK_AVAILABLE:
            self.viewer3D.render_optic()


class MatplotlibViewer(QWidget):
    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.current_theme = "dark"
        
        # --- Main Layout ---
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)

        # --- Toolbar Header ---
        self.toolbar_container = QWidget()
        self.toolbar_container.setObjectName("ViewerToolbarContainer")
        toolbar_layout = QHBoxLayout(self.toolbar_container)
        toolbar_layout.setContentsMargins(5, 0, 5, 0)
        self.layout.addWidget(self.toolbar_container)

        # --- Plot Canvas ---
        plot_container = QWidget()
        plot_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0,0,0,0)
        self.layout.addWidget(plot_container, 1)

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)

        # --- Add Toolbar ---
        # Using CustomMatplotlibToolbar from analysis_panel to disable coordinates in the bar
        self.toolbar = CustomMatplotlibToolbar(self.canvas, self.toolbar_container)
        toolbar_layout.addWidget(self.toolbar)
        
        # --- Cursor Coordinates Label ---
        self.cursor_coord_label = QLabel("", self.canvas)
        self.cursor_coord_label.setObjectName("CursorCoordLabel")
        self.cursor_coord_label.setStyleSheet("background-color:rgba(0,0,0,0.65);color:white;padding:2px 4px;border-radius:3px;")
        self.cursor_coord_label.setVisible(False)
        self.cursor_coord_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # --- Connections ---
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move_on_plot)

        # Initial plot
        self.plot_optic()
        
    def on_mouse_move_on_plot(self, event):
        """Handle mouse movement over the plot to display coordinates."""
        if event.inaxes:
            x_coord = f"{event.xdata:.3f}"
            y_coord = f"{event.ydata:.3f}"
            self.cursor_coord_label.setText(f"(Z, Y) = ({x_coord}, {y_coord})")
            self.cursor_coord_label.adjustSize()
            # Position at top-left corner with a small margin
            self.cursor_coord_label.move(5, 5)
            self.cursor_coord_label.setVisible(True)
            self.cursor_coord_label.raise_()
        else:
            self.cursor_coord_label.setVisible(False)

    def update_theme(self, theme="dark"):
        """Updates the plot's theme and replots."""
        if self.current_theme != theme:
            self.current_theme = theme
            self.plot_optic()

    def plot_optic(self):
        """Applies theme styles and completely redraws the optic."""
        # ** FIX: This is the crucial part **
        # Apply the current theme's styles every time we draw
        gui_plot_utils.apply_gui_matplotlib_styles(theme=self.current_theme)

        self.ax.clear()
        # Explicitly set the background color on the axes and figure
        face_color = matplotlib.rcParams['figure.facecolor']
        self.figure.set_facecolor(face_color)
        self.ax.set_facecolor(face_color)

        optic = self.connector.get_optic()
        if optic and optic.surface_group.num_surfaces > 0:
            try:
                rays2d_plotter = Rays2D(optic)
                system_plotter = OptilandOpticalSystemPlotter(optic, rays2d_plotter, projection="2d")
                rays2d_plotter.plot(self.ax, fields="all", wavelengths="primary", num_rays=3, distribution="line_y")
                system_plotter.plot(self.ax)
                self.ax.set_title(f"System: {optic.name} (2D)", color=matplotlib.rcParams['text.color'])
                self.ax.set_xlabel("Z-axis (mm)")
                self.ax.set_ylabel("Y-axis (mm)")
                self.ax.axis("equal")
                self.ax.grid(True, linestyle="--", alpha=0.7)
            except Exception as e:
                self.ax.text(0.5, 0.5, f"Error plotting Optiland 2D view:\n{e}", ha="center", va="center", transform=self.ax.transAxes, color='red')
                print(f"MatplotlibViewer Error: {e}")
        else:
            self.ax.text(0.5, 0.5, "2D Viewer (Matplotlib)\nNo Optic data or empty system.", ha="center", va="center", transform=self.ax.transAxes)

        self.canvas.draw()


class VTKViewer(QWidget):
    def __init__(self, connector: OptilandConnector, parent=None):
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
        self.iren.SetInteractorStyle(
            vtk.vtkInteractorStyleTrackballCamera()
        )  # Already default
        self.setup_default_camera()
        # self.render_optic() # Initial render done by connector signals
        self.iren.Initialize()

    def setup_default_camera(self):
        self.renderer.SetBackground(0.1, 0.2, 0.4)
        camera = self.renderer.GetActiveCamera()
        if (
            camera
        ):  # camera might not be initialized if GetRenderWindow isn't fully ready
            camera.SetPosition(0.2, 0, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 1, 0)
            self.renderer.ResetCamera()
            camera.Elevation(0)
            camera.Azimuth(150)

    def render_optic(self):
        if not VTK_AVAILABLE:
            return

        self.renderer.RemoveAllViewProps()  # Clear previous actors
        optic = self.connector.get_optic()

        if optic and optic.surface_group.num_surfaces > 0:
            try:
                # Similar to 2D, use Optiland's lower-level 3D plotters
                # OpticViewer3D.view() is blocking and manages its own window.
                rays3d_plotter = Rays3D(optic)
                system_plotter = OptilandOpticalSystemPlotter(
                    optic, rays3d_plotter, projection="3d"
                )

                # Plot rays (this also calculates extents for system_plotter)
                # Default parameters
                rays3d_plotter.plot(
                    self.renderer,
                    fields="all",
                    wavelengths="primary",
                    num_rays=24,
                    distribution="ring",
                )

                # Plot system components
                system_plotter.plot(self.renderer)

                # Set a nice default view if camera is available
                if (
                    not self.renderer.GetActiveCamera()
                ):  # if camera was not set up before
                    self.setup_default_camera()
                else:
                    self.renderer.ResetCameraClippingRange()
                    self.renderer.ResetCamera()

            except Exception as e:
                print(f"VTKViewer Error: {e}")
                # Optionally display error in the VTK window itself
                textActor = vtk.vtkTextActor()
                textActor.SetInput(f"Error rendering 3D view:\n{e}")
                textActor.GetTextProperty().SetColor(1, 0, 0)
                self.renderer.AddActor2D(textActor)
        else:
            # Placeholder if no optic data
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetRadius(0.1)  # Small sphere for empty
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
