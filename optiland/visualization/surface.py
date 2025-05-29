"""Surface Visualization Module

This module contains classes for visualizing optical surfaces in 2D and 3D.

Kramer Harrison, 2024
"""

import numpy as np
import vtk

import optiland.backend as be
from optiland.physical_apertures import RadialAperture
from optiland.rays import RealRays
from optiland.visualization.utils import revolve_contour, transform, transform_3d


class Surface2D:
    """A class used to represent a 2D surface for visualization.

    Args:
        surf (Surface): The surface object containing the geometry.
        extent (tuple): The extent of the surface in the x and y directions.

    Attributes:
        surf (Surface): The surface object containing the geometry.
        ray_extent (tuple): The extent of rays on the surface.

    Methods:
        plot(ax):
            Plots the surface on the given matplotlib axis.

    """

    def __init__(self, surface, ray_extent):
        self.surf = surface

        if self.surf.aperture:
            x_min, x_max, y_min, y_max = self.surf.aperture.extent
            self.extent = be.max(be.array([x_min, x_max, y_min, y_max]))
        else:
            self.extent = ray_extent

    def plot(self, ax):
        """Plots the surface on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the
                surface will be plotted.

        """
        x, y, z = self._compute_sag()

        # convert to global coordinates and return
        _, y, z = transform(x, y, z, self.surf, is_global=False)

        y = be.to_numpy(y)
        z = be.to_numpy(z)

        ax.plot(z, y, "gray")

    def _compute_sag(self):
        """Computes the sag of the surface in local coordinates and handles
        clipping due to physical apertures.

        Returns:
            tuple: A tuple containing arrays of x, y, and z coordinates.

        """
        # local coordinates
        x = be.zeros(128)
        y = be.linspace(-self.extent, self.extent, 128)
        z = self.surf.geometry.sag(x, y)

        # handle physical apertures
        if self.surf.aperture:
            y = be.copy(y)  # required to maintain gradient for torch backend
            intensity = be.ones_like(x)
            rays = RealRays(x, y, x, x, x, x, intensity, x)
            self.surf.aperture.clip(rays)
            y[rays.i == 0] = be.nan

        return x, y, z


class Surface3D(Surface2D):
    """A class used to represent a 3D surface for visualization.

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
        """Plots the surface on the given renderer.

        Args:
            renderer (vtkRenderer): The renderer to which the surface actor
                will be added.

        """
        actor = self.get_surface()
        self._configure_material(actor)
        renderer.AddActor(actor)

    def get_surface(self):
        """Retrieves the surface actor based on the symmetry of the surface
        geometry.

        If the surface geometry is symmetric, it retrieves a symmetric surface
        actor. Otherwise, it retrieves an asymmetric surface actor.

        Returns:
            actor: The surface actor, either symmetric or asymmetric, based on
                the surface geometry.

        """
        has_symmetric_aperture = (
            type(self.surf.aperture) is RadialAperture
            or self.surf.aperture is None  # "no aperture" is symmetric
        )
        is_symmetric = self.surf.geometry.is_symmetric
        if is_symmetric and has_symmetric_aperture:
            actor = self._get_symmetric_surface()
        else:
            actor = self._get_asymmetric_surface()
        actor = self._configure_material(actor)
        return actor

    def _get_symmetric_surface(self):
        """Generates a symmetric surface actor by computing the sag, revolving
        the contour, transforming it in 3D, and configuring its material
        properties.

        Returns:
            vtkActor: The configured 3D actor representing the symmetric
                surface.

        """
        x, y, z = self._compute_sag()
        actor = revolve_contour(x, y, z)
        actor = transform_3d(actor, self.surf)
        return actor

    def _get_asymmetric_surface(self):
        """Generates an asymmetric surface using Delaunay triangulation and
        returns a VTK actor for rendering.

        This method computes the 3D sag values, creates a VTK poly data object
        to store the points, applies Delaunay triangulation to generate a
        surface mesh, maps the surface to a VTK actor, configures the actor's
        material properties, and converts the actor to global coordinates.

        Returns:
            vtk.vtkActor: A VTK actor representing the asymmetric surface.

        """
        x, y, z = self._compute_sag_3d()
        x = be.to_numpy(x)
        y = be.to_numpy(y)
        z = be.to_numpy(z)

        # Apply aperture filtering to the grid of points
        if self.surf.aperture is not None:
            mask = self.surf.aperture.contains(x, y)
        else:
            r = np.hypot(x, y)
            mask = r <= be.to_numpy(self.extent)

        # Create VTK points.
        points = vtk.vtkPoints()
        num_rows, num_cols = x.shape

        # Map grid indices to point IDs
        point_ids = -np.ones((num_rows, num_cols), dtype=int)
        for i in range(num_rows):
            for j in range(num_cols):
                point_ids[i, j] = points.InsertNextPoint(x[i, j], y[i, j], z[i, j])

        # Create cells (quads) for the surface
        # Only include a quad if all four of its vertices lie inside aperture
        cells = vtk.vtkCellArray()
        for i in range(num_rows - 1):
            for j in range(num_cols - 1):
                # Check the four corners of the cell.
                if (
                    mask[i, j]
                    and mask[i + 1, j]
                    and mask[i + 1, j + 1]
                    and mask[i, j + 1]
                ):
                    quad = vtk.vtkQuad()
                    quad.GetPointIds().SetId(0, point_ids[i, j])
                    quad.GetPointIds().SetId(1, point_ids[i + 1, j])
                    quad.GetPointIds().SetId(2, point_ids[i + 1, j + 1])
                    quad.GetPointIds().SetId(3, point_ids[i, j + 1])
                    cells.InsertNextCell(quad)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Convert to global coordinates
        actor = transform_3d(actor, self.surf)

        return actor

    def _configure_material(self, actor):
        """Configures the material properties of a given actor.

        This method sets the color, ambient, diffuse, specular, and specular
        power properties of the actor's material.

        Args:
            actor: The actor whose material properties are to be configured.

        Returns:
            The actor with updated material properties.

        """
        actor.GetProperty().SetColor(1, 1, 1)
        actor.GetProperty().SetAmbient(0.5)
        actor.GetProperty().SetDiffuse(0.05)
        actor.GetProperty().SetSpecular(1.0)
        actor.GetProperty().SetSpecularPower(100)

        return actor

    def _compute_sag_3d(self):
        """Computes the 3D sag (surface height) of the optical surface within the
        given extent.

        This method calculates the sag of the optical surface over a 2D grid
        of points. The sag is computed using the surface's geometry.

        Returns:
            tuple: A tuple containing three numpy arrays (x, y, z)
                representing the coordinates of the points on the surface
                within the maximum radial extent.

        """
        if self.surf.aperture is not None:
            x_min, x_max, y_min, y_max = self.surf.aperture.extent
            x = be.linspace(x_min, x_max, 256)
            y = be.linspace(y_min, y_max, 256)
            x, y = be.meshgrid(x, y)
        else:
            x = be.linspace(-self.extent, self.extent, 256)
            x, y = be.meshgrid(x, x)

        z = self.surf.geometry.sag(x, y)
        return x, y, z
