"""Optiland - Spot Diagram Analysis Module

This module provides a spot diagram analysis for optical systems.

Kramer Harrison, 2024
"""
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


class SpotDiagram:
    """Spot diagram class

    This class generates data and plots real ray intersection locations
    on the final optical surface in an optical system. These plots
    are purely geometric and give an indication of the blur produced
    by aberrations in the system.

    Attributes:
        optic (optic.Optic): instance of the optic object to be assessed
        fields (tuple): fields at which data is generated
        wavelengths (tuple[float]): wavelengths at which data is generated
        num_rings (int): number of rings in pupil distribution for ray tracing
        data (List): contains spot data in a nested list. Data is ordered as
            field (dim 0), wavelength (dim 1), then x, y and intensity data
            (dim 2).
    """

    def __init__(self, optic, fields='all', wavelengths='all', num_rings=6,
                 distribution='hexapolar'):
        """Create an instance of SpotDiagram

        Note:
            The constructor also generates all data that may later be used for
            plotting

        Args:
            optic (optic.Optic): instance of the optic object to be assessed
            fields (tuple or str): fields at which data is generated.
                If 'all' is passed, then all field points are considered.
                Default is 'all'.
            wavelengths (str or tuple[float]): wavelengths at which data is
                generated. If 'all' is passed, then all wavelengths are
                considered. Default is 'all'.
            num_rings (int): number of rings in pupil distribution for ray
                tracing. Default is 6.
            distribution (str): pupil distribution type for ray tracing.
                Default is 'hexapolar'.

        Returns:
            None
        """
        self.optic = optic
        self.fields = fields
        self.wavelengths = wavelengths
        if self.fields == 'all':
            self.fields = self.optic.fields.get_field_coords()

        if self.wavelengths == 'all':
            self.wavelengths = self.optic.wavelengths.get_wavelengths()

        self.data = self._generate_data(self.fields, self.wavelengths,
                                        num_rings, distribution)

    def view(self, figsize=(12, 4)):
        """View the spot diagram

        Args:
            figsize (tuple): the figure size of the output window.
                Default is (12, 4).

        Returns:
            None
        """
        N = len(self.fields)
        num_rows = (N + 2) // 3

        fig, axs = plt.subplots(num_rows, 3,
                                figsize=(figsize[0], num_rows*figsize[1]),
                                sharex=True, sharey=True)
        axs = axs.flatten()

        # subtract centroid and find limits
        data = self._center_spots(deepcopy(self.data))
        geometric_size = self.geometric_spot_radius()
        axis_lim = np.max(geometric_size)

        # plot wavelengths for each field
        for k, field_data in enumerate(data):
            self._plot_field(axs[k], field_data, self.fields[k],
                             axis_lim, self.wavelengths)

        # remove empty axes
        for k in range(N, num_rows * 3):
            fig.delaxes(axs[k])

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')

        plt.tight_layout()
        plt.show()

    def centroid(self):
        """Centroid of each spot

        Returns:
            centroid (List): centroid for each field in the data.
        """
        norm_index = self.optic.wavelengths.primary_index
        centroid = []
        for field_data in self.data:
            centroid_x = np.mean(field_data[norm_index][0])
            centroid_y = np.mean(field_data[norm_index][1])
            centroid.append((centroid_x, centroid_y))
        return centroid

    def geometric_spot_radius(self):
        """Geometric spot radius of each spot

        Returns:
            geometric_size (List): Geometric spot radius for field and
                wavelength
        """
        data = self._center_spots(deepcopy(self.data))
        geometric_size = []
        for field_data in data:
            geometric_size_field = []
            for wave_data in field_data:
                r = np.sqrt(wave_data[0]**2 + wave_data[1]**2)
                geometric_size_field.append(np.max(r))
            geometric_size.append(geometric_size_field)
        return geometric_size

    def rms_spot_radius(self):
        """Root mean square (RMS) spot radius of each spot

        Returns:
            rms (List): RMS spot radius for each field and wavelength.
        """
        data = self._center_spots(deepcopy(self.data))
        rms = []
        for field_data in data:
            rms_field = []
            for wave_data in field_data:
                r2 = wave_data[0]**2 + wave_data[1]**2
                rms_field.append(np.sqrt(np.mean(r2)))
            rms.append(rms_field)
        return rms

    def _center_spots(self, data):
        """
        Centers the spots in the given data around their respective centroids.

        Args:
            data (List): A nested list representing the data containing spots.

        Returns:
            data (List): A nested list with the spots centered around their
                centroids.
        """
        centroids = self.centroid()
        data = deepcopy(self.data)
        for i, field_data in enumerate(data):
            for wave_data in field_data:
                wave_data[0] -= centroids[i][0]
                wave_data[1] -= centroids[i][1]
        return data

    def _generate_data(self, fields, wavelengths, num_rays=100,
                       distribution='hexapolar'):
        """
        Generate spot data for the given fields and wavelengths.

        Args:
            fields (List): A list of fields.
            wavelengths (List): A list of wavelengths.
            num_rays (int, optional): The number of rays to generate.
                Defaults to 100.
            distribution (str, optional): The distribution type.
                Defaults to 'hexapolar'.

        Returns:
            data (List): A nested list of spot intersection data for each
                field and wavelength.
        """
        data = []
        for field in fields:
            field_data = []
            for wavelength in wavelengths:
                field_data.append(self._generate_field_data(field,
                                                            wavelength,
                                                            num_rays,
                                                            distribution))
            data.append(field_data)
        return data

    def _generate_field_data(self, field, wavelength, num_rays=100,
                             distribution='hexapolar'):
        """
        Generates spot data for a given field and wavelength.

        Args:
            field (tuple): Tuple containing the field coordinates in (x, y).
            wavelength (float): The wavelength of the field.
            num_rays (int, optional): The number of rays to generate.
                Defaults to 100.
            distribution (str, optional): The distribution pattern of the
                rays. Defaults to 'hexapolar'.

        Returns:
            list: A list containing the x, y, and intensity values of the
                generated spot data.
        """
        self.optic.trace(*field, wavelength, num_rays, distribution)
        x = self.optic.surface_group.x[-1, :]
        y = self.optic.surface_group.y[-1, :]
        intensity = self.optic.surface_group.intensity[-1, :]
        return [x, y, intensity]

    def _plot_field(self, ax, field_data, field, axis_lim,
                    wavelengths, buffer=1.05):
        """
        Plot the field data on the given axis.

        Parameters:
            ax (matplotlib.axes.Axes): The axis to plot the field data on.
            field_data (list): List of tuples containing x, y, and intensity
                data points.
            field (tuple): Tuple containing the Hx and Hy field values.
            axis_lim (float): Limit of the x and y axis.
            wavelengths (list): List of wavelengths corresponding to the
                field data.
            buffer (float, optional): Buffer factor to extend the axis limits.
                Default is 1.05.

        Returns:
            None
        """
        markers = ['o', 's', '^']
        for k, points in enumerate(field_data):
            x, y, intensity = points
            mask = intensity != 0
            ax.scatter(x[mask], y[mask], s=10,
                       label=f'{wavelengths[k]:.4f} µm',
                       marker=markers[k % 3], alpha=0.7)
            ax.axis('square')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_xlim((-axis_lim*buffer, axis_lim*buffer))
            ax.set_ylim((-axis_lim*buffer, axis_lim*buffer))
        ax.set_title(f'Hx: {field[0]:.3f}, Hy: {field[1]:.3f}')
