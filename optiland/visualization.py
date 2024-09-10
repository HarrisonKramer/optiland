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
        _real_ray_extent: An array storing the maximum extent of rays for each
            surface.

    Methods:
        view: Visualizes the lenses and ray tracing.
        _plot_all_surfaces: Plots all the surfaces of the optical system.
        _plot_rays: Plots the rays for the given fields and wavelengths.
        _plot_lens: Plots a lens.
        _plot_surface: Plots a surface.
        _plot_line: Plots a line.
        _get_surface_extent: Returns the extent of a surface.
    """

    def __init__(self, optic):
        self.optic = optic

        n = self.optic.surface_group.num_surfaces
        self._real_ray_extent = np.zeros(n)

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
        _, self.ax = plt.subplots(figsize=figsize)
        self._plot_rays(fields=fields, wavelengths=wavelengths,
                        num_rays=num_rays, distribution=distribution)
        self._plot_all_surfaces()

        plt.axis('image')
        plt.show()

    def _plot_all_surfaces(self):
        """
        Plots all the surfaces of the optical system.
        """
        n = self.optic.n()

        for k in range(1, self.optic.surface_group.num_surfaces-1):
            surf = self.optic.surface_group.surfaces[k]
            if surf.is_reflective:
                y = self._get_surface_extent(k)
                z = surf.geometry.sag(y=y) + surf.geometry.cs.z
                self._plot_surface(y, z)

            if n[k] > 1:
                surf1 = self.optic.surface_group.surfaces[k]
                surf2 = self.optic.surface_group.surfaces[k+1]

                y1 = self._get_surface_extent(k)
                z1 = surf1.geometry.sag(y=y1) + surf1.geometry.cs.z

                y2 = self._get_surface_extent(k+1)
                z2 = surf2.geometry.sag(y=y2) + surf2.geometry.cs.z

                if n[k+1] == 1 and n[k-1] == 1:  # single lens
                    min_radius = min(np.min(y1), np.min(y2))
                    max_radius = max(np.max(y1), np.max(y2))

                elif n[k+1] != n[k] and n[k+1] != 1:  # bonded, 1st lens
                    y3 = np.linspace(-self._real_ray_extent[k+2],
                                     self._real_ray_extent[k+2], 128)
                    min_radius = min(np.min(y1), np.min(y2), np.min(y3))
                    max_radius = max(np.max(y1), np.max(y2), np.max(y3))

                elif n[k+1] != n[k] and n[k-1] != n[k]:  # bonded, 2nd lens
                    y0 = np.linspace(-self._real_ray_extent[k-1],
                                     self._real_ray_extent[k-1], 128)
                    min_radius = min(np.min(y0), np.min(y1), np.min(y2))
                    max_radius = max(np.max(y0), np.max(y1), np.max(y2))

                if y1[0] > min_radius:
                    y1 = np.insert(y1 + surf1.geometry.cs.y, 0, min_radius)
                    z1 = np.insert(z1, 0, z1[0])

                if y1[-1] < max_radius:
                    y1 = np.append(y1 + surf1.geometry.cs.y, max_radius)
                    z1 = np.append(z1, z1[-1])

                if y2[0] > min_radius:
                    y2 = np.insert(y2 + surf2.geometry.cs.y, 0, min_radius)
                    z2 = np.insert(z2, 0, z2[0])

                if y2[-1] < max_radius:
                    y2 = np.append(y2 + surf2.geometry.cs.y, max_radius)
                    z2 = np.append(z2, z2[-1])

                y = np.concatenate((y1, np.flip(y2)))
                z = np.concatenate((z1, np.flip(z2)))

                self._plot_lens(y, z)

        # plot image surface
        yi = np.linspace(-self._real_ray_extent[-1],
                         self._real_ray_extent[-1], 128)
        image_surf = self.optic.image_surface
        zi = image_surf.geometry.sag(y=yi) + image_surf.geometry.cs.z
        self._plot_surface(yi, zi)

    def _plot_rays(self, fields='all', wavelengths='primary', num_rays=3,
                   distribution='line_y'):
        """
        Plots the rays for the given fields and wavelengths.

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
                self.optic.trace(*field, wavelength, num_rays, distribution)
                x = self.optic.surface_group.x
                y = self.optic.surface_group.y
                z = self.optic.surface_group.z
                intensity = self.optic.surface_group.intensity

                # find maximum extent of rays
                for k in range(y.shape[0]):
                    if np.nanmax(np.abs(y[k, :])) > self._real_ray_extent[k]:
                        max_ray_height = np.nanmax(np.abs(y[k, :]))
                        self._real_ray_extent[k] = max_ray_height

                # if only one field, use different colors for each wavelength
                if len(fields) > 1:
                    color_idx = i
                else:
                    color_idx = j

                for k in range(z.shape[1]):
                    xk = x[:, k]
                    yk = y[:, k]
                    zk = z[:, k]
                    ik = intensity[:, k]

                    xk[ik == 0] = np.nan
                    zk[ik == 0] = np.nan
                    yk[ik == 0] = np.nan

                    self._plot_line(xk, yk, zk, f'C{color_idx}')

    def _plot_lens(self, y, z):
        """
        Plots a lens.

        Args:
            y: The y-coordinates of the lens profile.
            z: The z-coordinates of the lens profile.
        """
        vertices = np.column_stack((z, y))
        polygon = Polygon(vertices, closed=True, facecolor='lightgray',
                          edgecolor='gray')
        self.ax.add_patch(polygon)

    def _plot_surface(self, y, z):
        """
        Plots a surface.

        Args:
            y: The y-coordinates of the lens profile.
            z: The z-coordinates of the lens profile.
        """
        self.ax.plot(z, y, 'gray')

    def _plot_line(self, x, y, z, color):
        """
        Plots a line.

        Args:
            x: The x-coordinates of the line.
            y: The y-coordinates of the line.
            z: The z-coordinates of the line.
            color: The color of the line.
        """
        self.ax.plot(z, y, color, linewidth=1)

    def _get_surface_extent(self, surf_index):
        """
        Returns the extent of a surface.

        Args:
            surf_index: The index of the surface.

        Returns:
            The y-coordinates representing the extent of the surface.
        """
        x = np.zeros(256)
        y = np.linspace(-self._real_ray_extent[surf_index],
                        self._real_ray_extent[surf_index], 256)
        intensity = np.ones_like(x)
        surf = self.optic.surface_group.surfaces[surf_index]
        if surf.aperture:
            rays = RealRays(x, y, x, x, x, x, intensity, x)
            surf.aperture.clip(rays)
            y[rays.i == 0] = np.nan
        return y


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
        self.renWin = vtkRenderWindow()
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
        self.renWin.AddRenderer(self.renderer)

        self.iren.SetRenderWindow(self.renWin)

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

        self.renWin.SetSize(*figsize)
        self.renWin.SetWindowName('Optical System - 3D Viewer')
        self.renWin.Render()

        self.renderer.GetActiveCamera().SetPosition(1, 0, 0)
        self.renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
        self.renderer.GetActiveCamera().SetViewUp(0, 1, 0)
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Elevation(0)
        self.renderer.GetActiveCamera().Azimuth(150)

        self.renWin.Render()
        self.iren.Start()

    def _plot_lens(self, y, z, make_transparent=True):
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

    def _plot_surface(self, y, z):
        """
        Plot a surface in the 3D visualization.

        Args:
            y (list): The y-coordinates of the surface points.
            z (list): The z-coordinates of the surface points.
        """
        self._plot_lens(y, z, make_transparent=False)

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
