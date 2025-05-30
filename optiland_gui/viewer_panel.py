# optiland_gui/viewer_panel.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from PySide6.QtCore import Slot

# Matplotlib integration
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# VTK integration (optional, ensure VTK is installed)
try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

from .optiland_connector import OptilandConnector

class ViewerPanel(QWidget):
    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Viewer")

        self.layout = QVBoxLayout(self)
        self.tabWidget = QTabWidget()
        self.layout.addWidget(self.tabWidget)

        # 2D Viewer (Matplotlib)
        self.viewer2D = MatplotlibViewer(self.connector, self)
        self.tabWidget.addTab(self.viewer2D, "2D View")

        # 3D Viewer (VTK)
        self.viewer3D = VTKViewer(self.connector, self) if VTK_AVAILABLE else QLabel("VTK not available or not installed.")
        self.tabWidget.addTab(self.viewer3D, "3D View")

        self.connector.opticChanged.connect(self.update_viewers)

    @Slot()
    def update_viewers(self):
        self.viewer2D.plot()
        if VTK_AVAILABLE:
            self.viewer3D.update_scene()

class MatplotlibViewer(QWidget):
    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.ax = None
        self.plot() # Initial plot

    def plot(self):
        if self.ax:
            self.ax.clear()
        else:
            self.ax = self.figure.add_subplot(111)

        # Placeholder plot - Replace with Optiland's 2D drawing
        # optic = self.connector.get_optic()
        # if optic and hasattr(optic, 'draw'): # Check if Optiland's Optic has a draw method
        #     try:
        #         # Assuming optic.draw() can take an axes argument or OptilandViewer needs to be used
        #         # For now, a simple plot based on surface count
        #         num_surfaces = self.connector.get_surface_count()
        #         self.ax.plot(range(num_surfaces), [i**0.5 for i in range(num_surfaces)], 'ro-')
        #         self.ax.set_title(f"Optiland System: {optic.name} (2D Placeholder)")
        #     except Exception as e:
        #         self.ax.text(0.5, 0.5, f"Error plotting Optiland 2D view:\n{e}", ha='center', va='center')
        # else:
        self.ax.text(0.5, 0.5, "2D Viewer (Matplotlib)\nConnect to Optiland data",
                     ha='center', va='center', transform=self.ax.transAxes)

        self.ax.set_xlabel("Z-axis (mm)")
        self.ax.set_ylabel("Y-axis (mm)")
        self.ax.grid(True)
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

        self.setup_scene()
        self.iren.Initialize()

    def setup_scene(self):
        # Placeholder VTK scene - Replace with Optiland's 3D drawing
        self.renderer.Clear() # Clear previous actors

        # Example: Add a simple sphere
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(0.0, 0.0, 0.0)
        sphereSource.SetRadius(1.0)
        sphereSource.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0.1, 0.1) # Reddish

        self.renderer.AddActor(actor)
        self.renderer.SetBackground(0.1, 0.2, 0.4) # Dark blue background
        self.renderer.ResetCamera()

    @Slot()
    def update_scene(self):
        if not VTK_AVAILABLE:
            return
        # This method should be called when the Optic object changes
        # It would typically clear the renderer and add new actors based on the Optic data
        print("VTK Viewer: Updating scene (placeholder)")
        self.setup_scene() # Re-run setup for now
        self.vtkWidget.GetRenderWindow().Render()