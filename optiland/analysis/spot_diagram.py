"""Spot Diagram Analysis

This module provides a spot diagram analysis for optical systems.

Kramer Harrison, 2024
"""
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({'font.size': 12, 'font.family': 'cambria'})


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

    def view(self, figsize=(12, 4), add_airy_disk=False):
        """View the spot diagram

        Args:
            figsize (tuple): the figure size of the output window.
                Default is (12, 4).
            add_airy_disk (bool): Airy disc visualization controller.

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
                             axis_lim, self.wavelengths, add_airy_disk=add_airy_disk)

        # remove empty axes
        for k in range(N, num_rows * 3):
            fig.delaxes(axs[k])

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        plt.tight_layout()
        plt.show()

    def get_image_coordinates(self):
        """Generates the coordinate list on the image plane.

        Returns:
            coordinate_list (List): x, and y coordinates as float.
        """
        field_coords = self.optic.fields.get_field_coords()
        coordinate_list = []
        for idx in range(len(field_coords)):
            ray = self.optic.trace_generic(Hx = field_coords[idx][0], 
                                           Hy = field_coords[idx][1], 
                                           Px = 0, Py = 0, 
                                           wavelength=self.optic.wavelengths.primary_wavelength.value)
            coordinate_list.append([ray.x, ray.y])
        return coordinate_list
        

    def angle_from_cosine(self, cos1, cos2):
        """Calculate the angle (in radians and degrees) between two vectors given their cosine values.

        Returns:
            theta_rad (float): angle in radians.
        """
        # Compute the angle using arccos
        theta_rad = np.abs(np.arccos(cos1) % (np.pi/2)  - np.arccos(cos2)% (np.pi/2))
        #theta_deg = np.degrees(theta_rad)  # Convert to degrees

        return theta_rad

    def f_number(self, n, theta):
        """Calculates the F#

        Returns:
            N_w (float): Physical F number.
        """
        N_w = 1 / (2 * n * np.sin(theta))
        return N_w

    def airy_radius(self, n_w, wavelength):
        """
        Calculates the airy radius
        Returns:
            r (float): Airy radius.
        """
        r = 1.22 * n_w * wavelength
        return r
    
    def generate_marginal_rays(self, wavelength):
        """Generates marginal rays at the edges of the stop.
        
        Returns:
            ray_north, ray_south, ray_east, ray_west (RealRays tuple):  
        """
        ray_north = self.optic.trace_generic(Hx=0, Hy=0, Px=1, Py=0, wavelength=wavelength)
        ray_south = self.optic.trace_generic(Hx=0, Hy=0, Px=-1, Py=0, wavelength=wavelength)
        ray_east = self.optic.trace_generic(Hx=0, Hy=0, Px=0, Py=1, wavelength=wavelength)
        ray_west = self.optic.trace_generic(Hx=0, Hy=0, Px=0, Py=-1, wavelength=wavelength)

        return ray_north, ray_south, ray_east, ray_west

    # generate multiple chief ray's angle
    def generate_chief_rays_cosines(self, wavelength):
        """Generates the cosine values of chief rays for each field at the image plane.

        Returns:
            chief_ray_cosines_list (List): 2D list, each having (x, y) data of cosine value (direction vectors of the rays) at the image plane. 
        """
        coords = self.optic.fields.get_field_coords()
        chief_ray_cosines_list = []
        for coord in coords:
            H_x, H_y = coord[0], coord[1]
            ray_chief = self.optic.trace_generic(Hx=H_x, Hy=H_y, Px=0, Py=0, wavelength=wavelength) 
            chief_ray_cosines_x = ray_chief.L
            chief_ray_cosines_y = ray_chief.M
            chief_ray_cosines_list.append([chief_ray_cosines_x, chief_ray_cosines_y])
            
        return chief_ray_cosines_list
    def airy_disc_x_y(self, wavelength):

        """Generates marginal rays, chief rays, then compares the angle between them.
        Averaging each x and y axes, produces F# (N_w), then calculates the airy airy radius at each x-y axes.
        The procedure is done for each field defined by the user.

        Returns:
            airy_rad_tuple (tuple): A tuple containing arrays of airy radius at each x-y axis (r_x, r_y).
        """

        ray_north, ray_south, ray_east, ray_west =  self.generate_marginal_rays(wavelength)
        
        chief_ray_cosines_list = self.generate_chief_rays_cosines(wavelength)
        
        airy_rad_x_list = []
        airy_rad_y_list = []
        for chief_ray_cosines in chief_ray_cosines_list:
            # relative angles along x axis
            rel_angle_north = abs(self.angle_from_cosine(chief_ray_cosines[0], ray_north.L))
            rel_angle_south = abs(self.angle_from_cosine(chief_ray_cosines[0], ray_south.L))
            # relative angles along y axis
            rel_angle_east = abs(self.angle_from_cosine(chief_ray_cosines[1], ray_east.M))
            rel_angle_west = abs(self.angle_from_cosine(chief_ray_cosines[1], ray_west.M))
            
            avg_angle_x = (rel_angle_north + rel_angle_south) / 2
            avg_angle_y = (rel_angle_east + rel_angle_west) / 2
        
            N_w_x = self.f_number(n = 1, theta = avg_angle_x)
            N_w_y = self.f_number(n = 1, theta = avg_angle_y)
            
            airy_rad_x = self.airy_radius(N_w_x, wavelength)
            airy_rad_y = self.airy_radius(N_w_y, wavelength)
            airy_rad_x_list.append(airy_rad_x)
            airy_rad_y_list.append(airy_rad_y)
        airy_rad_tuple = (airy_rad_x_list, airy_rad_y_list)
        return airy_rad_tuple
    
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
                    wavelengths, buffer=1.05,  add_airy_disk=False):
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
            add_airy_disk (bool, optional): Whether to plot the airy disc or not, for analysis purposes.

        Returns:
            None
        """
        markers = ['o', 's', '^']
        coordinate_list = self.get_image_coordinates()  # final coordinate list on the image plane.
        if add_airy_disk:
            airy_rad_x, airy_rad_y = self.airy_disc_x_y(wavelength=self.optic.wavelengths.primary_wavelength.value)
            # Add the ellipse at the image planes.
            for k, coordinates in enumerate(coordinate_list):
                ellipse = patches.Ellipse((coordinates[0], coordinates[1]), width=airy_rad_x[k], height=airy_rad_y[k], linestyle = "--", edgecolor='black', fill=False, linewidth=1)
                ax.add_patch(ellipse)

            adjusted_axis_lim = max(axis_lim , max(airy_rad_x), max(airy_rad_y)) * buffer
        else:
            adjusted_axis_lim = axis_lim*buffer  # if airy disc is not present, prevent over buffering.

        for k, points in enumerate(field_data):
            x, y, intensity = points
            mask = intensity != 0
            ax.scatter(x[mask], y[mask], s=10,
                       label=f'{wavelengths[k]:.4f} µm',
                       marker=markers[k % 3], alpha=0.7)
            ax.axis('square')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_xlim((-adjusted_axis_lim, adjusted_axis_lim))
            ax.set_ylim((-adjusted_axis_lim, adjusted_axis_lim))

        ax.set_title(f'Hx: {field[0]:.3f}, Hy: {field[1]:.3f}')
        ax.grid(alpha=0.25)