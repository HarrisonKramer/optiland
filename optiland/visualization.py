# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
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
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from optiland.surfaces import ReflectiveSurface
from optiland.rays import RealRays


class LensViewer:
    # TODO: does not plot simple singlet correctly. Rays do not converge..
    # TODO: does not plot lenses with a single field correctly..

    def __init__(self, optic):
        self.optic = optic

        n = self.optic.surface_group.num_surfaces
        self._real_ray_extent = np.zeros(n)

    def view(self, fields='all', wavelengths='primary', num_rays=3,
             distribution='line_y', figsize=(10, 4)):
        _, ax = plt.subplots(figsize=figsize)
        self._plot_rays(ax, fields=fields, wavelengths=wavelengths,
                        num_rays=num_rays, distribution=distribution)
        self._plot_all_surfaces(ax)

        plt.axis('image')
        plt.show()

    def _plot_all_surfaces(self, ax):
        n = self.optic.n()

        for k in range(1, self.optic.surface_group.num_surfaces-1):
            surf = self.optic.surface_group.surfaces[k]
            if isinstance(surf, ReflectiveSurface):
                y = self._get_surface_extent(k)
                z = surf.geometry.sag(y=y) + surf.geometry.cs.z
                ax.plot(z, y, 'gray', linewidth=2)

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

                vertices = np.column_stack((z, y))
                polygon = Polygon(vertices, closed=True, facecolor='lightgray',
                                  edgecolor='gray')
                ax.add_patch(polygon)

        # plot image surface
        yi = np.linspace(-self._real_ray_extent[-1],
                         self._real_ray_extent[-1], 128)
        image_surf = self.optic.image_surface
        zi = image_surf.geometry.sag(y=yi) + image_surf.geometry.cs.z
        ax.plot(zi, yi, 'gray')

    def _plot_rays(self, ax, fields='all', wavelengths='primary', num_rays=3,
                   distribution='line_y'):
        if fields == 'all':
            fields = self.optic.fields.get_field_coords()

        if wavelengths == 'primary':
            wavelengths = [self.optic.wavelengths.primary_index]

        for i, field in enumerate(fields):
            for wavelength in wavelengths:
                self.optic.trace(*field, wavelength, num_rays, distribution)
                z = self.optic.surface_group.z
                y = self.optic.surface_group.y
                energy = self.optic.surface_group.energy

                # find maximum extent of rays
                for k in range(y.shape[0]):
                    if np.nanmax(np.abs(y[k, :])) > self._real_ray_extent[k]:
                        surf = self.optic.surface_group.surfaces[k]
                        max_ray_height = np.nanmax(np.abs(y[k, :]))
                        self._real_ray_extent[k] = max([max_ray_height,
                                                        surf.semi_aperture])

                for k in range(z.shape[1]):
                    zk = z[:, k]
                    yk = y[:, k]
                    ek = energy[:, k]

                    zk[ek == 0] = np.nan
                    yk[ek == 0] = np.nan

                    ax.plot(zk, yk, f'C{i}', linewidth=1)

    def _get_surface_extent(self, surf_index):
        x = np.zeros(256)
        y = np.linspace(-self._real_ray_extent[surf_index],
                        self._real_ray_extent[surf_index], 256)
        energy = np.ones_like(x)
        surf = self.optic.surface_group.surfaces[surf_index]
        if surf.aperture:
            rays = RealRays(x, y, x, x, x, x, energy, x)
            surf.aperture.clip(rays)
            y[rays.e == 0] = np.nan
        return y


class LensViewer3D(LensViewer):

    def __init__(self, optic):
        super().__init__(optic)

    def view(self, fields='all', wavelengths='primary', num_rays=2,
             distribution='hexapolar', figsize=(1200, 800)):
        self.renderer = vtkRenderer()
        renWin = vtkRenderWindow()
        renWin.AddRenderer(self.renderer)

        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        style = vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)

        self._real_ray_extent = self._plot_rays(fields, wavelengths, num_rays,
                                                distribution)
        self._plot_all_surfaces()

        self.renderer.GradientBackgroundOn()
        self.renderer.SetGradientMode(vtkViewport.GradientModes.VTK_GRADIENT_VERTICAL)
        self.renderer.SetBackground(0.7, 0.8, 0.9)
        self.renderer.SetBackground2(0.4, 0.5, 0.6)

        renWin.SetSize(*figsize)
        renWin.SetWindowName('Optical System - 3D Viewer')
        renWin.Render()

        self.renderer.GetActiveCamera().SetPosition(1, 0, 0)
        self.renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
        self.renderer.GetActiveCamera().SetViewUp(0, 1, 0)
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Elevation(0)
        self.renderer.GetActiveCamera().Azimuth(150)

        renWin.Render()
        iren.Start()

    def _plot_all_surfaces(self):
        # TODO: rewrite so that this function is the same for the 2D case
        n = self.optic.n()

        for k in range(1, self.optic.surface_group.num_surfaces-1):
            surf = self.optic.surface_group.surfaces[k]
            if isinstance(surf, ReflectiveSurface):
                y = self._get_surface_extent(k)
                z = surf.geometry.sag(y=y) + surf.geometry.cs.z
                self._plot_mirror(y, z)

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

    def _plot_rays(self, fields='all', wavelengths='primary', num_rays=2,
                   distribution='hexapolar'):
        if fields == 'all':
            fields = self.optic.fields.get_field_coords()

        if wavelengths == 'primary':
            wavelengths = [self.optic.wavelengths.primary_index]

        n = self.optic.surface_group.num_surfaces
        real_ray_extent = np.zeros(n)

        # Same colors as default matplotlib
        colors = [(0.122, 0.467, 0.706),
                  (1.000, 0.498, 0.055),
                  (0.173, 0.627, 0.173),
                  (0.839, 0.153, 0.157),
                  (0.580, 0.404, 0.741),
                  (0.549, 0.337, 0.294),
                  (0.890, 0.467, 0.761),
                  (0.498, 0.498, 0.498),
                  (0.737, 0.741, 0.133),
                  (0.090, 0.745, 0.812)]

        for i, field in enumerate(fields):
            for wavelength in wavelengths:
                self.optic.trace(*field, wavelength, num_rays, distribution)
                x = self.optic.surface_group.x
                y = self.optic.surface_group.y
                z = self.optic.surface_group.z

                # find maximum extent of rays
                for k in range(y.shape[0]):
                    if np.nanmax(np.abs(y[k, :])) > real_ray_extent[k]:
                        surf = self.optic.surface_group.surfaces[k]
                        max_ray_height = np.nanmax(np.abs(y[k, :]))
                        real_ray_extent[k] = max([max_ray_height,
                                                  surf.semi_aperture])

                for k in range(z.shape[1]):
                    for j in range(1, z.shape[0]):
                        p0 = [x[j-1, k], y[j-1, k], z[j-1, k]]
                        p1 = [x[j, k], y[j, k], z[j, k]]

                        self._plot_ray(p0, p1, color=colors[i % 10])

        return real_ray_extent

    def _plot_ray(self, p0, p1, color):
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

    def _plot_lens(self, y, z):
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
        surfaceActor.GetProperty().SetColor(0.9, 0.9, 1.0)
        surfaceActor.GetProperty().SetOpacity(0.5)
        surfaceActor.GetProperty().SetSpecular(1.0)
        surfaceActor.GetProperty().SetSpecularPower(50.0)

        self.renderer.AddActor(surfaceActor)
