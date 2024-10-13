import numpy as np
import vtk
from optiland.visualization.utils import transform
from optiland.rays import RealRays


class Surface2D:
    """
    A class used to represent a 2D surface for visualization.

    Args:
        surf (Surface): The surface object containing the geometry.
        extent (tuple): The extent of the surface in the x and y directions.

    Attributes:
        surf (Surface): The surface object containing the geometry.
        extent (tuple): The extent of the surface in the x and y directions.

    Methods:
        plot(ax):
            Plots the surface on the given matplotlib axis.
    """

    def __init__(self, surface, extent):
        self.surf = surface
        self.extent = extent

    def plot(self, ax):
        """
        Plots the surface on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the
                surface will be plotted.
        """
        x, y, z = self._compute_sag()

        # convert to global coordinates and return
        x, y, z = transform(x, y, z, self.surf, is_global=False)

        ax.plot(z, y, 'gray')

    def _compute_sag(self):
        """
        Computes the sag of the surface in local coordinates and handles
        clipping due to physical apertures.

        Returns:
            tuple: A tuple containing arrays of x, y, and z coordinates.
        """
        # local coordinates
        x = np.zeros(128)
        y = np.linspace(-self.extent[1], self.extent[1], 128)
        z = self.surf.geometry.sag(x, y)

        # handle physical apertures
        if self.surf.aperture:
            intensity = np.ones_like(x)
            rays = RealRays(x, y, x, x, x, x, intensity, x)
            self.surf.aperture.clip(rays)
            y[rays.i == 0] = np.nan

        return x, y, z


class Surface3D:
    """
    A class used to represent a 3D surface for visualization.

    Args:
        surf (Surface): The surface object containing the geometry.
        extent (tuple): The extent of the surface in the x and y directions.

    Attributes:
        surf (Surface): The surface object containing the geometry.
        extent (tuple): The extent of the surface in the x and y directions.

    Methods:
        plot(renderer):
            Plots the 3D surface using the provided VTK renderer.
    """

    def __init__(self, surface, extent):
        self.surf = surface
        self.extent = extent

    def plot(self, renderer):
        """
        Plots the 3D surface using the provided VTK renderer.

        Args:
            renderer (vtkRenderer): The VTK renderer to which the surface will
                be added.
        """
        x, y, z = self._compute_sag()

        # convert to global coordinates and return
        x, y, z = transform(x, y, z, self.surf, is_global=False)

        points = vtk.vtkPoints()
        for i in range(len(x)):
            points.InsertNextPoint(x[i], y[i], z[i])

        # Create a polydata object to store the points
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)

        # Apply Delaunay triangulation to generate surface mesh
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(poly_data)
        delaunay.Update()

        # Map the surface to a VTK actor for rendering
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(delaunay.GetOutput())

        # Configure the actor
        actor = vtk.vtkActor()
        self._configure_material(actor)
        actor.SetMapper(mapper)

        # Render the surface
        renderer.AddActor(actor)

    def _configure_material(self, actor):
        """
        Configures the material properties of the VTK actor.

        Args:
            actor (vtkActor): The VTK actor whose material properties will be
                configured.
        """
        actor.GetProperty().SetColor(1, 1, 1)
        actor.GetProperty().SetAmbient(0.5)
        actor.GetProperty().SetDiffuse(0.1)
        actor.GetProperty().SetSpecular(1.0)
        actor.GetProperty().SetSpecularPower(100)

    def _compute_sag(self):
        """
        Computes the sag (z-coordinates) of the surface based on its geometry.

        Returns:
            tuple: Three numpy arrays representing the x, y, and z coordinates
                of the surface points.
        """
        r_max = np.hypot(self.extent[0], self.extent[1])

        x = np.linspace(-r_max, r_max, 128)
        x, y = np.meshgrid(x, x)
        z = self.surf.geometry.sag(x, y)
        r = np.hypot(x, y)

        x = x[r <= r_max]
        y = y[r <= r_max]
        z = z[r <= r_max]

        return x, y, z
