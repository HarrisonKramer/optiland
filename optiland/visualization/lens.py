"""Lens Visualization Module

This module contains classes for visualizing lenses in 2D and 3D.

Kramer Harrison, 2024
"""

import numpy as np
import vtk
from matplotlib.patches import Polygon

import optiland.backend as be
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
                         Each element is expected to be an object (e.g., Surface3D)
                         that has a `surf` attribute (the actual Surface object)
                         and an `extent` attribute.

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
            sags = self._compute_sag()  # sags are in global coordinates
            self._plot_lenses(renderer, sags)
        else:
            self._plot_surfaces(renderer)
            self._plot_surface_edges(renderer)

    def _plot_single_lens(self, renderer, x, y, z):
        """Plots a single lens by revolving a contour and configuring its
        material. The input coordinates are expected to be in global space.

        Args:
            renderer (vtkRenderer): The renderer to which the lens actor will
                be added.
            x (numpy.ndarray): The x-coordinates of the contour points.
            y (numpy.ndarray): The y-coordinates of the contour points.
            z (numpy.ndarray): The z-coordinates of the contour points.

        """
        # Points are already global from _compute_sag with apply_transform=True
        actor = revolve_contour(be.to_numpy(x), be.to_numpy(y), be.to_numpy(z))
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
        prop = actor.GetProperty()
        prop.SetOpacity(0.5)
        prop.SetColor(0.9, 0.9, 1.0)
        prop.SetSpecular(1.0)
        prop.SetSpecularPower(50.0)
        return actor

    def _plot_surfaces(self, renderer):
        """Plots the non-symmetric surfaces of the lens in the given renderer.

        This method retrieves a pre-transformed actor for each surface,
        configures its material, and adds it to the renderer.
        If a surface's extent is less than the maximum lens extent,
        an annulus is plotted to fill the gap.

        Args:
            renderer (vtkRenderer): The VTK renderer where the surfaces will
                be added.

        """
        max_extent = self._get_max_extent()
        for (
            surface_3d_obj
        ) in self.surfaces:  # surface_3d_obj is an instance of e.g. Surface3D
            actor = (
                surface_3d_obj.get_surface()
            )  # retrieves actor from Surface3D (already transformed)
            actor = self._configure_material(actor)
            renderer.AddActor(actor)

            # Add annulus if surface extent does not extend to lens edge
            if surface_3d_obj.extent < max_extent:
                self._plot_annulus(surface_3d_obj, renderer)  # Pass renderer

    def _plot_annulus(self, surface_3d_obj, renderer):
        """Plots a VTK annulus for a given surface.

        The annulus extends from the surface's current extent to the maximum
        lens extent. It is defined in the local coordinate system of the
        surface and then transformed to its global position.

        Args:
            surface_3d_obj: The surface object (e.g., an instance of Surface3D)
                            for which to plot the annulus. It should have `surf`
                            (Surface object with geometry and cs) and `extent`
                            attributes.
            renderer (vtkRenderer): The VTK renderer to add the annulus actor to.
        """
        surf_props = surface_3d_obj.surf  # Actual Surface object with .geometry
        surf_geom = surf_props.geometry

        inner_radius = surface_3d_obj.extent
        outer_radius = self._get_max_extent()

        if outer_radius <= inner_radius:
            # No annulus needed if the surface already extends to or beyond max_extent
            return

        num_theta_points = 64
        # Do not take endpoint that overlaps with first point
        theta = be.linspace(0, 2 * be.pi, num_theta_points + 1)[:-1]

        # --- Define annulus geometry in local coordinates ---
        # Inner circle points
        x_inner_local = inner_radius * be.cos(theta)
        y_inner_local = inner_radius * be.sin(theta)
        z_inner_local = surf_geom.sag(x_inner_local, y_inner_local)

        # Outer circle points
        x_outer_local = outer_radius * be.cos(theta)
        y_outer_local = outer_radius * be.sin(theta)
        z_outer_local = surf_geom.sag(x_outer_local, y_outer_local)

        # Convert to NumPy arrays for VTK
        points_data = [
            be.to_numpy(arr)
            for arr in [
                x_inner_local,
                y_inner_local,
                z_inner_local,
                x_outer_local,
                y_outer_local,
                z_outer_local,
            ]
        ]
        x_inner_np, y_inner_np, z_inner_np, x_outer_np, y_outer_np, z_outer_np = (
            points_data
        )

        # Create vtkPoints object
        vtk_pts = vtk.vtkPoints()
        for i in range(num_theta_points):
            vtk_pts.InsertNextPoint(x_inner_np[i], y_inner_np[i], z_inner_np[i])
        for i in range(num_theta_points):
            vtk_pts.InsertNextPoint(x_outer_np[i], y_outer_np[i], z_outer_np[i])

        # Create vtkCellArray for the quadrilaterals forming the annulus surface
        quads = vtk.vtkCellArray()
        for i in range(num_theta_points):
            quad = vtk.vtkQuad()
            p_inner_curr = i
            p_inner_next = (i + 1) % num_theta_points
            # Offset for outer circle points in vtk_pts array
            p_outer_curr = num_theta_points + i
            p_outer_next = num_theta_points + ((i + 1) % num_theta_points)

            quad.GetPointIds().SetId(0, p_inner_curr)
            quad.GetPointIds().SetId(1, p_inner_next)
            quad.GetPointIds().SetId(2, p_outer_next)
            quad.GetPointIds().SetId(3, p_outer_curr)
            quads.InsertNextCell(quad)

        # Create vtkPolyData
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_pts)
        polydata.SetPolys(quads)

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        annulus_actor = vtk.vtkActor()
        annulus_actor.SetMapper(mapper)

        # Configure material properties
        annulus_actor = self._configure_material(annulus_actor)
        annulus_actor = transform_3d(annulus_actor, surf_props)
        renderer.AddActor(annulus_actor)

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

    def _get_edge_points(self, surface_obj):
        """Computes the (x, y, z) local coordinates of the edges of the lens.

        Args:
            surface_obj (Surface): The surface object itself (not Surface3D).

        Returns:
            tuple: A tuple containing arrays of x, y, and z
                coordinates in the local coordinate system of surface_obj.
        """
        max_extent_lens = self._get_max_extent()
        theta = be.linspace(0, 2 * be.pi, 256)  # 256 points for smooth edge

        x_local = max_extent_lens * be.cos(theta)
        y_local = max_extent_lens * be.sin(theta)
        z_local = surface_obj.surf.geometry.sag(x_local, y_local)

        return x_local, y_local, z_local
