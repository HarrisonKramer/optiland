import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from optiland.surfaces import ReflectiveSurface


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
                y = np.linspace(-self._real_ray_extent[k],
                                self._real_ray_extent[k], 128)
                z = surf.geometry.sag(y=y) + surf.geometry.cs.z
                ax.plot(z, y, 'gray', linewidth=2)

            if n[k] > 1:
                surf1 = self.optic.surface_group.surfaces[k]
                surf2 = self.optic.surface_group.surfaces[k+1]

                y1 = np.linspace(-self._real_ray_extent[k],
                                 self._real_ray_extent[k], 128)
                z1 = surf1.geometry.sag(y=y1) + surf1.geometry.cs.z

                y2 = np.linspace(-self._real_ray_extent[k+1],
                                 self._real_ray_extent[k+1], 128)
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

                # find maximum extent of real of paraxial rays
                for k in range(y.shape[0]):
                    if np.max(np.abs(y[k, :])) > self._real_ray_extent[k]:
                        surf = self.optic.surface_group.surfaces[k]
                        max_ray_height = np.max(np.abs(y[k, :]))
                        self._real_ray_extent[k] = max([max_ray_height,
                                                        surf.semi_aperture])

                for k in range(z.shape[1]):
                    ax.plot(z[:, k], y[:, k], f'C{i}', linewidth=1)
