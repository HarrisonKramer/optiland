"""Optiland Distribution Module

This module provides visualization tools for optical systems using VTK and
Matplotlib. It includes the `LensViewer` class, which allows for the
visualization of lenses, rays, and their interactions within an optical system.
The module supports plotting rays with different distributions, wavelengths,
and through various fields of view. It also visualizes the surfaces of the
optical elements, providing insights into the design and performance of the
system.

Kramer Harrison, 2023
"""
import os
import warnings
import vtkmodules.vtkRenderingOpenGL2  # noqa
from vtkmodules.vtkFiltersSources import vtkLineSource
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData,
    vtkLine
)
from vtkmodules.vtkFiltersModeling import vtkRotationalExtrusionFilter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkViewport
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from optiland.rays import RealRays
from optiland import geometries, materials


class LensViewer:
    """
    A class for visualizing optical systems and traced rays.

    Args:
        optic: An instance of the `Optic` class representing the optical
            system.

    Attributes:
        optic: An instance of the `Optic` class representing the optical
            system.
        ax: The matplotlib axis object.

    Methods:
        view: Visualizes the lenses and ray tracing.
    """

    def __init__(self, optic):
        self.optic = optic
        self._is_symmetric = self._is_rotationally_symmetric()

        self._num_surfaces = self.optic.surface_group.num_surfaces
        self._lens_group_id = np.zeros(self._num_surfaces, dtype=int)
        self.ax = None

        self._points_gcs = []
        self._points_lcs = []
        self._sags_gcs = None

    def view(self, fields='all', wavelengths='primary', num_rays=3,
             distribution='line_y', figsize=(10, 4)):
        """
        Visualizes the lenses and traced rays.

        Args:
            fields: The fields at which to trace the rays. Default is 'all'.
            wavelengths: The wavelengths at which to trace the rays.
                Default is 'primary'.
            num_rays: The number of rays to trace for each field and
                wavelength. Default is 3.
            distribution: The distribution of the rays. Default is 'line_y'.
            figsize: The size of the figure. Default is (10, 4).
        """
        if not self._is_symmetric:
            warnings.warn('The optical system is not rotationally symmetric. '
                          'The visualization may not be accurate.')

        _, self.ax = plt.subplots(figsize=figsize)

        self._draw_rays(fields=fields, wavelengths=wavelengths,
                        num_rays=num_rays, distribution=distribution)

        self._generate_surfaces()
        self._draw_object_surface()
        self._draw_image_surface()
        self._draw_reflective_surfaces()
        self._draw_lenses()

        plt.show()

    def _is_rotationally_symmetric(self):
        """Check if the optical system is rotationally symmetric."""
        for surf in self.optic.surface_group.surfaces:
            if not surf.is_rotationally_symmetric():
                return False
        return True

    def _draw_rays(self, fields='all', wavelengths='primary', num_rays=3,
                   distribution='line_y'):
        """
        Draw the rays for the given fields and wavelengths.

        Args:
            fields: The fields at which to trace the rays. Default is 'all'.
            wavelengths: The wavelengths at which to trace the rays.
                Default is 'primary'.
            num_rays: The number of rays to trace for each field and
                wavelength. Default is 3.
            distribution: The distribution of the rays. Default is 'line_y'.
        """
        if fields == 'all':
            fields = self.optic.fields.get_field_coords()

        if wavelengths == 'primary':
            wavelengths = [self.optic.wavelengths.primary_wavelength.value]

        for i, field in enumerate(fields):
            for j, wavelength in enumerate(wavelengths):

                # trace rays
                points_gcs, intensity = self._trace_rays(
                    field, wavelength, num_rays, distribution
                    )

                # convert to surface-local coordinates
                points_lcs = self._transform(points_gcs, is_global=True)

                # record rays
                self._record_rays(points_gcs, points_lcs)

                # if only one field, use different colors for each wavelength
                if len(fields) > 1:
                    color_idx = i
                else:
                    color_idx = j

                # plot lines
                self._draw_lines(points_gcs, intensity, color_idx)

    def _trace_rays(self, field, wavelength, num_rays, distribution):
        """
        Traces rays through the optical system and returns their coordinates
        and intensities.

        Args:
            field (tuple): A tuple representing the field coordinates.
            wavelength (float): The wavelength of the rays.
            num_rays (int): The number of rays to trace.
            distribution (str): The distribution type of the rays.

        Returns:
            tuple: A tuple containing:
                - points (numpy.ndarray): A 2D array with the coordinates of
                    the rays.
                - intensity (numpy.ndarray): An array with the intensities of
                    the rays.
        """
        self.optic.trace(*field, wavelength, num_rays, distribution)
        x = self.optic.surface_group.x
        y = self.optic.surface_group.y
        z = self.optic.surface_group.z
        intensity = self.optic.surface_group.intensity

        points = np.stack([x, y, z], axis=0)

        return points, intensity

    def _transform(self, points, is_global=True):
        """
        Transforms the given points based on the surfaces in the optic's
        surface group.

        Args:
            points (tuple): A tuple containing three numpy arrays (x, y, z)
                representing the coordinates of the points.
            is_global (bool, optional): If True, transforms points to the
                local coordinate system of each surface. If False, transforms
                points to the global coordinate system. Defaults to True.
        Returns:
            numpy.ndarray: A 3D numpy array with the transformed coordinates.
        """
        x, y, z = points
        x_new = x.copy()
        y_new = y.copy()
        z_new = z.copy()

        t = np.zeros(x.shape[0])

        for i, surf in enumerate(self.optic.surface_group.surfaces):
            ray_origins = RealRays(x[i, :], y[i, :], z[i, :], t, t, t, t, t)
            if is_global:
                surf.geometry.localize(ray_origins)
            else:
                surf.geometry.globalize(ray_origins)
            x_new[i, :] = ray_origins.x
            y_new[i, :] = ray_origins.y
            z_new[i, :] = ray_origins.z

        return np.stack([x_new, y_new, z_new], axis=0)

    def _record_rays(self, points_gcs, points_lcs):
        """
        Records the given ray points in global coordinate system (GCS) and
        local coordinate system (LCS).

        Args:
            points_gcs (list or array-like): Points in the global coordinate
                system.
            points_lcs (list or array-like): Points in the local coordinate
                system.
        """
        self._points_gcs.append(points_gcs)
        self._points_lcs.append(points_lcs)

    def _draw_lines(self, points, intensity, color_idx):
        """
        Draws lines based on the given points and intensity.

        Args:
            points (tuple): A tuple containing three numpy arrays (x, y, z)
                representing the coordinates of the points.
            intensity (numpy.ndarray): A 2D numpy array representing the
                intensity of each point.
            color_idx (int): An index representing the matploltib color to be
                used for drawing the lines, e.g., 0 for 'C0', 1 for 'C1', etc.
        """
        x, y, z = points

        # loop through rays
        for k in range(z.shape[1]):
            xk = x[:, k]
            yk = y[:, k]
            zk = z[:, k]
            ik = intensity[:, k]

            # remove rays outside aperture
            xk[ik == 0] = np.nan
            zk[ik == 0] = np.nan
            yk[ik == 0] = np.nan

            self._draw_single_line(xk, yk, zk, f'C{color_idx}')

    def _draw_single_line(self, x, y, z, color):
        """
        Draws a single line on the plot.

        Args:
            x (array-like): The x-coordinates of the points.
            y (array-like): The y-coordinates of the points.
            z (array-like): The z-coordinates of the points.
            color (str): The color of the line.
        """
        self.ax.plot(z, y, color, linewidth=1)

    def _generate_surfaces(self):
        """
        Generates the surfaces for the optical system.

        This method performs the following steps:
        1. Determines the extent of each surface by calling
            `_get_surface_extents`.
        2. Generates the sags (surface heights) of all surfaces using
            `_generate_sags`.
        3. Converts the sags to global coordinates using `_transform`.
        """
        # determine extent of each surface
        ray_extents, surf_extents = self._get_surface_extents()

        # generate sags of all surfaces
        sags = self._generate_sags(ray_extents, surf_extents)

        # convert sags to global coordinates
        self._sags_gcs = self._transform(sags, is_global=False)

    def _get_surface_extents(self):
        """
        Calculate the extents of rays and lens surfaces.

        This method computes the maximum radial extents of rays on each surface
        and determines the extents of lens surfaces.

        Returns:
            tuple: A tuple containing:
            - ray_extents (np.ndarray): The maximum radial extents of rays on
                each surface.
            - surf_extents (np.ndarray): The extents of the physical lens
                surface.
        """
        # find ray extents on each surface
        r_max_list = []
        for points in self._points_lcs:
            x, y, _ = points
            r = np.hypot(x, y)
            r_max = np.max(r, axis=1)
            r_max_list.append(r_max)

        r_max_list = np.array(r_max_list)
        ray_extents = np.max(r_max_list, axis=0)

        # find lens extents
        n = self.optic.n()
        surf_extents = ray_extents.copy()
        for k in range(self._num_surfaces):
            if n[k] > 1:
                if n[k+1] == 1 and n[k-1] == 1:  # single lens
                    surf_extents[k:k+2] = np.max(surf_extents[k:k+2])

                elif n[k+1] != n[k] and n[k+1] != 1:  # bonded, 1st lens
                    surf_extents[k:k+3] = np.max(surf_extents[k:k+3])

                elif n[k+1] != n[k] and n[k-1] != n[k]:  # bonded, 2nd lens
                    surf_extents[k-1:k+2] = np.max(surf_extents[k-1:k+2])

        return ray_extents, surf_extents

    def _generate_sags(self, ray_extents, surf_extents):
        """
        Generate the sag values for each surface in the optical system.

        This method computes the sag values for each surface in the optical
        system based on the provided ray extents and surface extents. It
        handles the extension of surfaces if necessary and accounts for
        physical apertures.

        Args:
            ray_extents (list of float): The extents of the rays for each
                surface.
            surf_extents (list of float): The physical extents of the surfaces.

        Returns:
            numpy.ndarray: A 3D array of shape (3, num_surfaces, 128)
                containing the x, y, and z coordinates of the points on each
                surface.
        """
        # 128 points per surface
        points = np.zeros((3, self._num_surfaces, 128))
        x = np.zeros(128)
        for i, surf in enumerate(self.optic.surface_group.surfaces):

            # extend surface if necessary
            if ray_extents[i] < surf_extents[i]:
                y = np.linspace(-ray_extents[i], ray_extents[i], 126)
                r_max = np.array([surf_extents[i]])
                y = np.concatenate([-r_max, y, r_max])
            else:
                y = np.linspace(-ray_extents[i], ray_extents[i], 128)

            z = surf.geometry.sag(x, y)

            # handle physical apertures
            if surf.aperture:
                intensity = np.ones_like(x)
                rays = RealRays(x, y, x, x, x, x, intensity, x)
                surf.aperture.clip(rays)
                y[rays.i == 0] = np.nan

            points[0, i, :] = x
            points[1, i, :] = y
            points[2, i, :] = z

        return points

    def _plot_surface(self, x, y, z):
        """
        Plots a surface.

        Args:
            y: The y-coordinates of the lens profile.
            z: The z-coordinates of the lens profile.
        """
        self.ax.plot(z, y, 'gray')

    def _draw_object_surface(self):
        """
        Draws the surface of the first object in the surface group.

        This method retrieves the first surface from the `surface_group` of
        the `optic` object. If the surface is infinite, it extracts the x, y,
        and z coordinates from `_sags_gcs` and plots the surface using the
        `_plot_surface` method.
        """
        obj_surf = self.optic.surface_group.surfaces[0]
        if obj_surf.is_infinite:
            x = self._sags_gcs[0, 0, :]
            y = self._sags_gcs[1, 0, :]
            z = self._sags_gcs[2, 0, :]
            self._plot_surface(x, y, z)

    def _draw_image_surface(self):
        """
        Draws the image surface of the optic system.

        This method retrieves the last surface from the optic's surface group
        and plots it if the surface's geometry is not at infinity. It uses the
        sags in the global coordinate system (GCS) to get the x, y, and z
        coordinates for plotting.
        """
        img_surf = self.optic.surface_group.surfaces[-1]
        if not np.isinf(img_surf.geometry.cs.z):
            x = self._sags_gcs[0, -1, :]
            y = self._sags_gcs[1, -1, :]
            z = self._sags_gcs[2, -1, :]
            self._plot_surface(x, y, z)

    def _draw_reflective_surfaces(self):
        """
        Draws reflective surfaces in the optical system.

        This method iterates through the surfaces in the optical system's
        surface group. For each reflective surface, it extracts the x, y, and
        z coordinates from the global coordinate system (GCS) sags and plots
        the surface.
        """
        for i, surf in enumerate(self.optic.surface_group.surfaces):
            if surf.is_reflective:
                x = self._sags_gcs[0, i, :]
                y = self._sags_gcs[1, i, :]
                z = self._sags_gcs[2, i, :]
                self._plot_surface(x, y, z)

    def _draw_lenses(self):
        """
        Draws the lenses of the optical system.

        This method iterates through the surfaces of the optical system and
        draws each lens by concatenating the coordinates of the lens surfaces
        and passing them to the `_draw_single_lens` method.
        """
        n = self.optic.n()
        for i, surf in enumerate(self.optic.surface_group.surfaces):
            if n[i] > 1:
                x1 = self._sags_gcs[0, i, :]
                y1 = self._sags_gcs[1, i, :]
                z1 = self._sags_gcs[2, i, :]

                x2 = self._sags_gcs[0, i+1, :]
                y2 = self._sags_gcs[1, i+1, :]
                z2 = self._sags_gcs[2, i+1, :]

                x = np.concatenate([x1, x2[::-1]])
                y = np.concatenate([y1, y2[::-1]])
                z = np.concatenate([z1, z2[::-1]])

                self._draw_single_lens(x, y, z)

    def _draw_single_lens(self, x, y, z):
        """
        Draws a single lens on the plot.

        This method creates a polygon representing a lens using the provided
        coordinates and adds it to the plot.

        Args:
            x (float): The x-coordinate of the lens.
            y (array-like): The y-coordinates of the lens vertices.
            z (array-like): The z-coordinates of the lens vertices.
        """
        vertices = np.column_stack((z, y))
        polygon = Polygon(vertices, closed=True, facecolor='lightgray',
                          edgecolor='gray')
        self.ax.add_patch(polygon)


class LensViewer3D(LensViewer):
    """
    A class for visualizing optical systems in 3D.

    Args:
        optic (OpticalSystem): The optical system to visualize.

    Attributes:
        _rgb_colors (list): A list of RGB colors used for visualization.

    Methods:
        view: Visualize the optical system in 3D.
        _plot_lens: Plot a lens in the 3D visualization.
        _plot_surface: Plot a surface in the 3D visualization.
        _plot_line: Plot a line in the 3D visualization.
    """

    def __init__(self, optic):
        super().__init__(optic)
        self.ren_win = vtkRenderWindow()
        self.iren = vtkRenderWindowInteractor()

        # matplotlib default colors converted to RGB
        self._rgb_colors = [(0.122, 0.467, 0.706),
                            (1.000, 0.498, 0.055),
                            (0.173, 0.627, 0.173),
                            (0.839, 0.153, 0.157),
                            (0.580, 0.404, 0.741),
                            (0.549, 0.337, 0.294),
                            (0.890, 0.467, 0.761),
                            (0.498, 0.498, 0.498),
                            (0.737, 0.741, 0.133),
                            (0.090, 0.745, 0.812)]

    def view(self, fields='all', wavelengths='primary', num_rays=2,
             distribution='hexapolar', figsize=(1200, 800)):
        """
        Visualize the optical system in 3D.

        Args:
            fields (str or list, optional): The fields to visualize.
                Defaults to 'all'.
            wavelengths (str or list, optional): The wavelengths to visualize.
                Defaults to 'primary'.
            num_rays (int, optional): The number of rays (or rings) to trace.
                Defaults to 2.
            distribution (str, optional): The distribution of rays.
                Defaults to 'hexapolar'.
            figsize (tuple, optional): The size of the figure.
                Defaults to (1200, 800).
        """
        self.renderer = vtkRenderer()
        self.ren_win.AddRenderer(self.renderer)

        self.iren.SetRenderWindow(self.ren_win)

        style = vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        self._plot_rays(fields, wavelengths, num_rays, distribution)
        self._plot_all_surfaces()

        self.renderer.GradientBackgroundOn()
        self.renderer.SetGradientMode(
            vtkViewport.GradientModes.VTK_GRADIENT_VERTICAL
        )

        self.renderer.SetBackground(0.8, 0.9, 1.0)
        self.renderer.SetBackground2(0.4, 0.5, 0.6)

        self.ren_win.SetSize(*figsize)
        self.ren_win.SetWindowName('Optical System - 3D Viewer')
        self.ren_win.Render()

        self.renderer.GetActiveCamera().SetPosition(1, 0, 0)
        self.renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
        self.renderer.GetActiveCamera().SetViewUp(0, 1, 0)
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Elevation(0)
        self.renderer.GetActiveCamera().Azimuth(150)

        self.ren_win.Render()
        self.iren.Start()

    def _plot_lens(self, x, y, z, make_transparent=True):
        """
        Plot a lens in the 3D visualization.

        Args:
            y (list): The y-coordinates of the lens points.
            z (list): The z-coordinates of the lens points.
            make_transparent (bool, optional): Whether to make the lens
                transparent. Defaults to True.
        """
        pts = [(0, yi, zi) for yi, zi in zip(y, z)]

        points = vtkPoints()
        lines = vtkCellArray()
        for pt in pts:
            pt_id = points.InsertNextPoint(pt)
            if pt_id < len(pts) - 1:
                line = vtkLine()
                line.GetPointIds().SetId(0, pt_id)
                line.GetPointIds().SetId(1, pt_id + 1)
                lines.InsertNextCell(line)

        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)

        revolution = vtkRotationalExtrusionFilter()
        revolution.SetInputData(polydata)
        revolution.SetResolution(256)

        surfaceMapper = vtkPolyDataMapper()
        surfaceMapper.SetInputConnection(revolution.GetOutputPort())

        surfaceActor = vtkActor()
        surfaceActor.SetMapper(surfaceMapper)
        if make_transparent:
            surfaceActor.GetProperty().SetOpacity(0.5)

        surfaceActor.GetProperty().SetColor(0.9, 0.9, 1.0)
        surfaceActor.GetProperty().SetSpecular(1.0)
        surfaceActor.GetProperty().SetSpecularPower(50.0)

        self.renderer.AddActor(surfaceActor)

    def _plot_surface(self, x, y, z):
        """
        Plot a surface in the 3D visualization.

        Args:
            y (list): The y-coordinates of the surface points.
            z (list): The z-coordinates of the surface points.
        """
        self._plot_lens(x, y, z, make_transparent=False)

    def _plot_line(self, x, y, z, color):
        """
        Plot a line in the 3D visualization.

        Args:
            x (list): The x-coordinates of the line points.
            y (list): The y-coordinates of the line points.
            z (list): The z-coordinates of the line points.
            color (str): The color of the line.
        """
        color = self._rgb_colors[int(color[1:]) % 10]
        for k in range(1, len(x)):
            p0 = [x[k-1], y[k-1], z[k-1]]
            p1 = [x[k], y[k], z[k]]

            lineSource = vtkLineSource()
            lineSource.SetPoint1(p0)
            lineSource.SetPoint2(p1)

            lineMapper = vtkPolyDataMapper()
            lineMapper.SetInputConnection(lineSource.GetOutputPort())
            lineActor = vtkActor()
            lineActor.SetMapper(lineMapper)
            lineActor.GetProperty().SetLineWidth(1)
            lineActor.GetProperty().SetColor(color)

            self.renderer.AddActor(lineActor)


class LensInfoViewer:
    """
    A class for viewing information about a lens.

    Args:
        optic (Optic): The optic object containing the lens information.

    Attributes:
        optic (Optic): The optic object containing the lens information.

    Methods:
        view(): Prints the lens information in a tabular format.
    """
    def __init__(self, optic):
        self.optic = optic

    def view(self):
        """
        Prints the lens information in a tabular format.

        The lens information includes the surface type, radius, thickness,
        material, conic, and semi-aperture of each surface.
        """
        self.optic.update_paraxial()

        surf_type = []
        for surf in self.optic.surface_group.surfaces:
            if isinstance(surf.geometry, geometries.EvenAsphere):
                surf_type.append('Even Asphere')
            elif isinstance(surf.geometry, geometries.Plane):
                surf_type.append('Planar')
            elif isinstance(surf.geometry, geometries.StandardGeometry):
                surf_type.append('Standard')
            else:
                raise ValueError('Unknown surface type')

            if surf.is_stop:
                surf_type[-1] = 'Stop - ' + surf_type[-1]

        radii = self.optic.surface_group.radii
        thicknesses = np.diff(self.optic.surface_group.positions.ravel(),
                              append=np.nan)
        conic = self.optic.surface_group.conic
        semi_aperture = [surf.semi_aperture
                         for surf in self.optic.surface_group.surfaces]

        mat = []
        for surf in self.optic.surface_group.surfaces:
            if surf.is_reflective:
                mat.append('Mirror')
            elif isinstance(surf.material_post, materials.Material):
                mat.append(surf.material_post.name)
            elif isinstance(surf.material_post, materials.MaterialFile):
                mat.append(os.path.basename(surf.material_post.filename))
            elif surf.material_post.index == 1:
                mat.append('Air')
            elif isinstance(surf.material_post, materials.IdealMaterial):
                mat.append(surf.material_post.index)
            else:
                raise ValueError('Unknown material type')

        self.optic.update_paraxial()

        df = pd.DataFrame({
            'Type': surf_type,
            'Radius': radii,
            'Thickness': thicknesses,
            'Material': mat,
            'Conic': conic,
            'Semi-aperture': semi_aperture
        })
        print(df.to_markdown(headers='keys', tablefmt='psql'))
