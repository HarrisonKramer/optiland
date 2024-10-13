import numpy as np
from matplotlib.patches import Polygon
from optiland.visualization.utils import transform


class Lens2D:
    """
    A class to represent a 2D lens and provide methods for plotting it.

    Args:
        surfaces (list): A list of surface objects that make up the lens.

    Attributes:
        surfaces (list): A list of surface objects that make up the lens.

    Methods:
        plot(ax):
            Plots the lens on the given matplotlib axis.
    """

    def __init__(self, surfaces):
        self.surfaces = surfaces

    def plot(self, ax):
        """
        Plots the lens on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the
                lens will be plotted.
        """
        sags = self._compute_sag()
        self._plot_lenses(ax, sags)

    def _compute_sag(self):
        """
        Computes the sag of the lens in local coordinates and handles
        clipping due to physical apertures.

        Returns:
            list: A list of tuples containing arrays of x, y, and z
                coordinates.
        """
        max_extents = self._get_max_extent()
        sags = []
        for surf in self.surfaces:
            x, y, z = surf._compute_sag()

            # extend surface to max extent
            if surf.extent[0] < max_extents[0] or \
               surf.extent[1] < max_extents[1]:
                x, y, z = self._extend_surface(x, y, z, max_extents)

            # convert to global coordinates
            x, y, z = transform(x, y, z, surf.surf, is_global=False)

            sags.append((x, y, z))

        return sags

    def _get_max_extent(self):
        """
        Gets the maximum extent of all surfaces in the lens.

        Returns:
            numpy.ndarray: An array containing the maximum extent in the x
                and y directions.
        """
        extents = np.array([surf.extent for surf in self.surfaces])
        return np.nanmax(extents, axis=0)

    def _extend_surface(self, x, y, z, extent):
        """
        Extends the surface to the maximum extent.

        Args:
            x (numpy.ndarray): The x coordinates of the surface.
            y (numpy.ndarray): The y coordinates of the surface.
            z (numpy.ndarray): The z coordinates of the surface.
            extent (numpy.ndarray): The maximum extent of the surface.

        Returns:
            tuple: A tuple containing the extended x, y, and z coordinates.
        """
        x_max = np.array([extent[0]])
        y_max = np.array([extent[1]])

        x = np.concatenate([-x_max, x, x_max])
        y = np.concatenate([-y_max, y, y_max])
        z = np.concatenate([np.array([z[0]]), z, np.array([z[-1]])])

        return x, y, z

    def _plot_single_lens(self, ax, x, y, z):
        """
        Plot a single lens on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the
                lens will be plotted.
            x (numpy.ndarray): The x coordinates of the lens.
            y (numpy.ndarray): The y coordinates of the lens.
            z (numpy.ndarray): The z coordinates of the lens.
        """
        vertices = np.column_stack((z, y))
        polygon = Polygon(vertices, closed=True,
                          facecolor=(0.8, 0.8, 0.8, 0.6),
                          edgecolor=(0.5, 0.5, 0.5))
        ax.add_patch(polygon)

    def _plot_lenses(self, ax, sags):
        """
        Plot the lenses on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the
                lenses will be plotted.
            sags (list): A list of tuples containing arrays of x, y, and z
                coordinates for each surface.
        """
        for k in range(len(sags)-1):
            x1, y1, z1 = sags[k]
            x2, y2, z2 = sags[k+1]

            # plot lens
            x = np.concatenate([x1, x2[::-1]])
            y = np.concatenate([y1, y2[::-1]])
            z = np.concatenate([z1, z2[::-1]])

            self._plot_single_lens(ax, x, y, z)


class Lens3D(Lens2D):

    def __init__(self, surfaces):
        super().__init__(surfaces)

    def plot(self, renderer):
        pass

    def _plot_single_lens(self, renderer, x, y, z):
        pass
