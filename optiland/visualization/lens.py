"""Lens Visualization Module

This module contains classes for visualizing lenses in 2D and 3D.

Kramer Harrison, 2024
"""

import numpy as np
import vtk
from matplotlib.patches import Polygon

from optiland.visualization.utils import revolve_contour, transform, transform_3d


class Lens2D:
    """A class to represent a 2D lens and provide methods for plotting it.

    Args:
        surfaces (list): A list of surface objects that make up the lens.

    Attributes:
        surfaces (list): A list of surface objects that make up the lens.

    Methods:
        plot(ax):
            Plots the lens on the given matplotlib axis.

    """

    def __init__(self, surfaces):
        # TODO: raise warning when lens surfaces overlap
        self.surfaces = surfaces

    def plot(self, ax):
        """Plots the lens on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the
                lens will be plotted.

        """
        sags = self._compute_sag()
        self._plot_lenses(ax, sags)

    def _compute_sag(self, apply_transform=True):
        """Computes the sag of the lens in local coordinates and handles
        clipping due to physical apertures.

        Returns:
            list: A list of tuples containing arrays of x, y, and z
                coordinates.

        """
        max_extent = self._get_max_extent()
        sags = []
        for surf in self.surfaces:
            x, y, z = surf._compute_sag()

            # extend surface to max extent
            if surf.extent < max_extent:
                x, y, z = self._extend_surface(x, y, z, surf.surf, max_extent)

            # convert to global coordinates
            if apply_transform:
                x, y, z = transform(x, y, z, surf.surf, is_global=False)

            sags.append((x, y, z))

        return sags

    def _get_max_extent(self):
        """Gets the maximum radial extent of all surfaces in the lens in global
        coordinates.

        Returns:
            float: The maximum radial extent of all surfaces in the lens.

        """
        extents = np.array([surf.extent for surf in self.surfaces])
        return np.nanmax(extents, axis=0)

    def _extend_surface(self, x, y, z, surface, extent):
        """Extends the surface to the maximum extent.

        Args:
            x (numpy.ndarray): The x coordinates of the surface.
            y (numpy.ndarray): The y coordinates of the surface.
            z (numpy.ndarray): The z coordinates of the surface.
            surface (Surface): The surface object.
            extent (numpy.ndarray): The maximum extent of the surface.

        Returns:
            tuple: A tuple containing the extended x, y, and z coordinates.

        """
        y_new = np.array([extent])
        x = np.concatenate([np.array([0]), x, np.array([0])])
        y = np.concatenate([-y_new, y, y_new])
        z = np.concatenate([np.array([z[0]]), z, np.array([z[-1]])])

        surface.extent = extent

        return x, y, z

    def _plot_single_lens(self, ax, x, y, z):
        """Plot a single lens on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the
                lens will be plotted.
            x (numpy.ndarray): The x coordinates of the lens.
            y (numpy.ndarray): The y coordinates of the lens.
            z (numpy.ndarray): The z coordinates of the lens.

        """
        vertices = np.column_stack((z, y))
        polygon = Polygon(
            vertices,
            closed=True,
            facecolor=(0.8, 0.8, 0.8, 0.6),
            edgecolor=(0.5, 0.5, 0.5),
        )
        ax.add_patch(polygon)

    def _plot_lenses(self, ax, sags):
        """Plot the lenses on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the
                lenses will be plotted.
            sags (list): A list of tuples containing arrays of x, y, and z
                coordinates for each surface.

        """
        for k in range(len(sags) - 1):
            x1, y1, z1 = sags[k]
            x2, y2, z2 = sags[k + 1]

            # plot lens
            x = np.concatenate([x1, x2[::-1]])
            y = np.concatenate([y1, y2[::-1]])
            z = np.concatenate([z1, z2[::-1]])

            self._plot_single_lens(ax, x, y, z)


class Lens3D(Lens2D):
    """A class used to represent a 3D Lens, inheriting from Lens2D.

    Args:
        surfaces (list): A list of surfaces that make up the lens.

    Attributes:
        surfaces (list): A list of surfaces that make up the lens.

    Methods:
        is_symmetric:
            Checks if the lens is rotationally symmetric.
        plot(renderer):
            Plots the lens using the given VTK renderer.

    """

    def __init__(self, surfaces):
        super().__init__(surfaces)

    @property
    def is_symmetric(self):
        """Check if all surfaces in the lens are symmetric.

        This method iterates through each surface in the lens and checks if the
        geometry of the surface is symmetric. A surface is considered symmetric
        if its geometry's `is_symmetric` attribute is True and both `rx` and
        `ry` attributes of its coordinate system (`cs`) are zero.

        Returns:
            bool: True if all surfaces are symmetric, False otherwise.

        """
        for surf in self.surfaces:
            geometry = surf.surf.geometry
            if not geometry.is_symmetric:
                return False
            if (
                geometry.cs.rx != 0
                or geometry.cs.ry != 0
                or geometry.cs.x != 0
                or geometry.cs.y != 0
            ):
                return False
        return True

    def plot(self, renderer):
        """Plots the lens or surfaces using the provided renderer.

        Args:
            renderer: The rendering engine used to plot the lens or surfaces.

        """
        if self.is_symmetric:
            sags = self._compute_sag()
            self._plot_lenses(renderer, sags)
        else:
            self._plot_surfaces(renderer)
            self._plot_surface_edges(renderer)

    def _plot_single_lens(self, renderer, x, y, z):
        """Plots a single lens by revolving a contour and configuring its
        material.

        Args:
            renderer (vtkRenderer): The renderer to which the lens actor will
                be added.
            x (list of float): The x-coordinates of the contour points.
            y (list of float): The y-coordinates of the contour points.
            z (list of float): The z-coordinates of the contour points.

        """
        actor = revolve_contour(x, y, z)
        actor = self._configure_material(actor)
        renderer.AddActor(actor)

    def _configure_material(self, actor):
        """Configures the material properties of a given VTK actor.
        This method sets the opacity, color, specular, and specular power
        properties of the provided VTK actor to predefined values.

        Args:
            actor (vtkActor): The VTK actor whose material properties are to
                be configured.

        Returns:
            vtkActor: The VTK actor with updated material properties.

        """
        actor.GetProperty().SetOpacity(0.5)
        actor.GetProperty().SetColor(0.9, 0.9, 1.0)
        actor.GetProperty().SetSpecular(1.0)
        actor.GetProperty().SetSpecularPower(50.0)

        return actor

    def _plot_surfaces(self, renderer):
        """Plots the surfaces of the lens in the given renderer.

        This method computes the sag values for each surface, creates a 3D
        actor for each surface by revolving the contour, applies the necessary
        transformations, configures the material properties, and adds the
        actor to the renderer.

        Args:
            renderer (vtkRenderer): The VTK renderer where the surfaces will
                be added.

        """
        sags = self._compute_sag(apply_transform=False)
        for k, (x, y, z) in enumerate(sags):
            surface = self.surfaces[k].surf
            actor = revolve_contour(x, y, z)
            actor = transform_3d(actor, surface)
            actor = self._configure_material(actor)
            renderer.AddActor(actor)

    def _get_edge_surface(self, circle1, circle2):
        """Generates a VTK actor representing the surface between two circles.

        Args:
            circle1 (list of tuple): List of points representing the first
                circle.
            circle2 (list of tuple): List of points representing the second
                circle.

        Returns:
            vtk.vtkActor: VTK actor representing the surface between the two
                circles.

        """
        num_points = len(circle1)

        # Create vtkPoints object to hold all the points from both circles
        points = vtk.vtkPoints()
        for p in circle1:
            points.InsertNextPoint(p)
        for p in circle2:
            points.InsertNextPoint(p)

        # Create vtkCellArray to hold the quadrilaterals forming the surface
        polys = vtk.vtkCellArray()

        # Loop over the points and connect points from the two circles
        for i in range(num_points):
            next_i = (i + 1) % num_points  # Wrap around for the last point

            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, i)
            quad.GetPointIds().SetId(1, next_i)
            quad.GetPointIds().SetId(2, num_points + next_i)
            quad.GetPointIds().SetId(3, num_points + i)

            polys.InsertNextCell(quad)

        # Create the vtkPolyData object
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polys)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor = self._configure_material(actor)

        return actor

    def _plot_surface_edges(self, renderer):
        """Plots the edges of surfaces in a 3D renderer.

        This method generates circular edges for each surface. It then
        transforms these edges into the appropriate coordinate system and adds
        them to the renderer.

        Args:
            renderer: The 3D renderer object where the surface edges will be
                added.

        """
        circles = []
        for surface in self.surfaces:
            x, y, z = self._get_edge_points(surface)
            x, y, z = transform(x, y, z, surface.surf, is_global=False)
            circles.append(np.stack((x, y, z), axis=-1))

        for k in range(len(circles) - 1):
            circle1 = circles[k]
            circle2 = circles[k + 1]
            actor = self._get_edge_surface(circle1, circle2)
            renderer.AddActor(actor)

    def _get_edge_points(self, surface):
        """Computes the (x, y, z) local coordinates of the edges of the lens.

        Args:
            surface (Surface): The surface object.

        Returns:
            list: A list of tuples containing arrays of x, y, and z
                coordinates.

        """
        max_extent = self._get_max_extent()
        theta = np.linspace(0, 2 * np.pi, 256)

        x = max_extent * np.cos(theta)
        y = max_extent * np.sin(theta)
        z = surface.surf.geometry.sag(x, y)

        return x, y, z
