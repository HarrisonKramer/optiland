"""Lens Visualization Module

This module contains classes for visualizing lenses in 2D and 3D.

Kramer Harrison, 2024
"""

import numpy as np
import vtk
from matplotlib.patches import Polygon

import optiland.backend as be
from optiland.physical_apertures import RadialAperture
from optiland.visualization.utils import transform

from .surface import Surface3D


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
        extents = be.array([surf.extent for surf in self.surfaces])
        return be.nanmax(extents, axis=0)

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
        y_new = be.array([extent])
        x = be.concatenate([be.array([0]), x, be.array([0])])
        y = be.concatenate([-y_new, y, y_new])
        z = be.concatenate([be.array([z[0]]), z, be.array([z[-1]])])

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
        vertices = be.to_numpy(be.column_stack((z, y)))
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
            x = be.concatenate([x1, be.flip(x2)])
            y = be.concatenate([y1, be.flip(y2)])
            z = be.concatenate([z1, be.flip(z2)])

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

        self.plotting_surfaces_3d = []
        for surface_obj in self.surfaces:
            s_extent = None
            if hasattr(surface_obj, "aperture") and surface_obj.aperture is not None:
                if isinstance(surface_obj.aperture, RadialAperture):
                    s_extent = surface_obj.aperture.r_max
                elif hasattr(
                    surface_obj.aperture, "extent"
                ):  # For other aperture types like Rectangular
                    # Assuming aperture.extent gives (x_min, x_max, y_min, y_max)
                    x_min, x_max, y_min, y_max = surface_obj.aperture.extent
                    s_extent = be.max(be.abs(be.array([x_min, x_max, y_min, y_max])))

            if (
                s_extent is None
                and hasattr(surface_obj, "semi_aperture")
                and surface_obj.semi_aperture is not None
                and (
                    isinstance(surface_obj.semi_aperture, (float, int))
                    or (
                        hasattr(surface_obj.semi_aperture, "shape")
                        and not surface_obj.semi_aperture.shape
                    )
                )
            ):
                # Fallback to the surface's own semi_aperture if it's a simple scalar
                s_extent = surface_obj.semi_aperture

            if s_extent is None:
                # If no specific extent can be found for this surface,
                # we might need a default or a lens-system-wide max.
                # For now, let's use a default. This should be logged or improved later
                # if surfaces without clear extents are common.
                s_extent = 10.0  # Default extent if none found
                # Consider adding: print(f"Warning: Using default extent for surface
                # {surface_obj}")

            self.plotting_surfaces_3d.append(Surface3D(surface_obj, s_extent))

    @property
    def is_symmetric(self):
        """Check if all surfaces in the lens are rotationally symmetric.

        This method iterates through each underlying optical surface in the lens
        and checks if its geometry is intrinsically symmetric and if its
        coordinate system is aligned (no rotations or decenters relative to
        the optical axis).

        Returns:
            bool: True if all surfaces are symmetric and aligned, False otherwise.
        """
        for surf in self.surfaces:  # These are optiland.surface.Surface instances
            geometry = surf.geometry  # Corrected access to the geometry object
            if not geometry.is_symmetric:
                return False
            # Check coordinate system properties for tilts and decenters
            if (
                hasattr(geometry, "cs")  # Ensure cs attribute exists
                and (
                    geometry.cs.rx != 0
                    or geometry.cs.ry != 0
                    or geometry.cs.x != 0
                    or geometry.cs.y != 0
                )
            ):
                return False
        return True

    def plot(self, renderer):
        """Plots the lens surfaces using the provided VTK renderer.

        This method iterates through the `Surface3D` representations of each
        surface in the lens, allowing each surface to determine its own
        symmetric or asymmetric rendering. It then plots the edges connecting
        these surfaces.

        Args:
            renderer (vtkRenderer): The VTK renderer where the lens will be
                plotted.
        """
        # Both symmetric and asymmetric lenses are plotted by iterating
        # through their Surface3D representations.
        for surface_3d_obj in self.plotting_surfaces_3d:
            actor = (
                surface_3d_obj.get_surface()
            )  # Surface3D.get_surface decides symm/asymm
            renderer.AddActor(actor)

        # Plot the edges that connect these surfaces.
        self._plot_surface_edges(renderer)

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
        """Plots the asymmetric surfaces of the lens in the given renderer
        using Surface3D objects.
        """
        for surface_3d_obj in self.plotting_surfaces_3d:
            actor = surface_3d_obj.get_surface()
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
            x = be.to_numpy(x)
            y = be.to_numpy(y)
            z = be.to_numpy(z)
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
        theta = be.linspace(0, 2 * be.pi, 256)

        x = max_extent * be.cos(theta)
        y = max_extent * be.sin(theta)
        z = surface.surf.geometry.sag(x, y)

        return x, y, z
