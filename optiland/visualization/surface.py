import numpy as np
import vtk
from optiland.rays import RealRays
from optiland.visualization.utils import (
    transform,
    transform_3d,
    revolve_contour
)


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
        y = np.linspace(-self.extent, self.extent, 128)
        z = self.surf.geometry.sag(x, y)

        # handle physical apertures
        if self.surf.aperture:
            intensity = np.ones_like(x)
            rays = RealRays(x, y, x, x, x, x, intensity, x)
            self.surf.aperture.clip(rays)
            y[rays.i == 0] = np.nan

        return x, y, z


class Surface3D(Surface2D):
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
        super().__init__(surface, extent)

    def plot(self, renderer):
        """
        Plots the surface on the given renderer.

        Args:
            renderer (vtkRenderer): The renderer to which the surface actor
                will be added.
        """
        actor = self.get_surface()
        self._configure_material(actor)
        renderer.AddActor(actor)

    def get_surface(self):
        """
        Retrieves the surface actor based on the symmetry of the surface
        geometry.

        If the surface geometry is symmetric, it retrieves a symmetric surface
        actor. Otherwise, it retrieves an asymmetric surface actor.

        Returns:
            actor: The surface actor, either symmetric or asymmetric, based on
                the surface geometry.
        """
        if self.surf.geometry.is_symmetric:
            actor = self._get_symmetric_surface()
        else:
            actor = self._get_asymmetric_surface()
        return actor

    def _get_symmetric_surface(self):
        """
        Generates a symmetric surface actor by computing the sag, revolving
        the contour, transforming it in 3D, and configuring its material
        properties.

        Returns:
            vtkActor: The configured 3D actor representing the symmetric
                surface.
        """
        x, y, z = self._compute_sag()
        actor = revolve_contour(x, y, z)
        actor = transform_3d(actor, self.surf)
        actor = self._configure_material(actor)
        return actor

    def _get_asymmetric_surface(self):
        """
        Generates an asymmetric surface using Delaunay triangulation and
        returns a VTK actor for rendering.

        This method computes the 3D sag values, creates a VTK poly data object
        to store the points, applies Delaunay triangulation to generate a
        surface mesh, maps the surface to a VTK actor, configures the actor's
        material properties, and converts the actor to global coordinates.

        Returns:
            vtk.vtkActor: A VTK actor representing the asymmetric surface.
        """
        x, y, z = self._compute_sag_3d()

        points = vtk.vtkPoints()
        for i in range(len(x)):
            points.InsertNextPoint(x[i], y[i], z[i])

        # Create a poly_data object to store the points
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
        actor.SetMapper(mapper)
        actor = self._configure_material(actor)

        # Convert to global coordinates
        actor = transform_3d(actor, self.surf)

        return actor

    def _configure_material(self, actor):
        """
        Configures the material properties of a given actor.

        This method sets the color, ambient, diffuse, specular, and specular
        power properties of the actor's material.

        Args:
            actor: The actor whose material properties are to be configured.

        Returns:
            The actor with updated material properties.
        """
        actor.GetProperty().SetColor(1, 1, 1)
        actor.GetProperty().SetAmbient(0.5)
        actor.GetProperty().SetDiffuse(0.1)
        actor.GetProperty().SetSpecular(1.0)
        actor.GetProperty().SetSpecularPower(100)

        return actor

    def _compute_sag_3d(self):
        """
        Computes the 3D sag (surface height) of the optical surface within the
        given extent.

        This method calculates the sag of the optical surface over a 2D grid
        of points within the maximum radial extent defined by the object's
        extent attribute. The sag is computed using the surface's geometry.

        Returns:
            tuple: A tuple containing three numpy arrays (x, y, z)
                representing the coordinates of the points on the surface
                within the maximum radial extent.
        """
        x = np.linspace(-self.extent, self.extent, 128)
        x, y = np.meshgrid(x, x)
        z = self.surf.geometry.sag(x, y)
        r = np.hypot(x, y)

        in_aperture = r <= self.r_extent
        x = x[in_aperture]
        y = y[in_aperture]
        z = z[in_aperture]

        return x, y, z
