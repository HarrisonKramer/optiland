"""Spot Diagram Analysis

This module provides a spot diagram analysis for optical systems.

Kramer Harrison, 2024
"""

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib import patches

import optiland.backend as be
from optiland.visualization.utils import transform

from .base import BaseAnalysis


@dataclass
class SpotData:
    """Stores the x, y coordinates and intensity of a spot.

    Attributes:
        x: Array of x-coordinates.
        y: Array of y-coordinates.
        intensity: Array of intensity values.
    """

    x: be.array
    y: be.array
    intensity: be.array


class SpotDiagram(BaseAnalysis):
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
        data (List[List[SpotData]]): contains spot data in a nested list.
            Data is ordered as field (dim 0), wavelength (dim 1), then
            SpotData objects.
        coordinates (Literal['global', 'local']): Coordinate system for data
                                                  and plotting.

    """

    def __init__(
        self,
        optic,
        fields="all",
        wavelengths="all",
        num_rings=6,
        distribution="hexapolar",
        coordinates: Literal["global", "local"] = "local",
    ):
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
            coordinates (Literal['global', 'local'], optional): Coordinate system
                for data generation and plotting. Defaults to "local".

        """
        if fields == "all":
            self.fields = optic.fields.get_field_coords()
        else:
            self.fields = fields

        if coordinates not in ["global", "local"]:
            raise ValueError("Coordinates must be 'global' or 'local'.")
        self.coordinates = coordinates

        self.num_rings = num_rings
        self.distribution = distribution

        super().__init__(optic, wavelengths)

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

        fig, axs = plt.subplots(
            num_rows,
            3,
            figsize=(figsize[0], num_rows * figsize[1]),
            sharex=True,
            sharey=True,
        )
        axs = axs.flatten()

        # Subtract centroid and find limits
        data = self._center_spots(self.data)
        geometric_size = self.geometric_spot_radius()
        # stack into an (N_fields, N_waves) array/tensor, then reduce:
        # first convert each row into a 1D tensor/array
        rows = [be.stack(row, axis=0) for row in geometric_size]
        # now stack rows into a 2D
        gs_array = be.stack(rows, axis=0)
        # take the global max
        axis_lim = be.max(gs_array)

        if add_airy_disk:
            wavelength = self.optic.wavelengths.primary_wavelength.value
            centroids = self.centroid()
            chief_ray_centers = self.generate_chief_rays_centers(wavelength=wavelength)
            airy_rad_x, airy_rad_y = self.airy_disc_x_y(wavelength=wavelength)

        # Do not calculate airy disc parameters if not required.
        else:
            wavelength = None
            centroids = None
            chief_ray_centers = None
            airy_rad_x, airy_rad_y = None, None

        # Plot wavelengths for each field
        for k, field_data in enumerate(data):
            # Calculate the real centroid difference for the current field for
            # airy disc
            if add_airy_disk:
                real_centroid_x = chief_ray_centers[k][0] - centroids[k][0]
                real_centroid_y = chief_ray_centers[k][1] - centroids[k][1]
            else:
                real_centroid_x, real_centroid_y = None, None
            self._plot_field(
                axs[k],
                field_data,
                self.fields[k],
                axis_lim,
                self.wavelengths,
                add_airy_disk=add_airy_disk,
                field_index=k,
                airy_rad_x=airy_rad_x,
                airy_rad_y=airy_rad_y,
                real_centroid_x=real_centroid_x,
                real_centroid_y=real_centroid_y,
            )

        # Remove empty axes
        for k in range(N, num_rows * 3):
            fig.delaxes(axs[k])

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        plt.tight_layout()
        plt.show()

    def angle_from_cosine(self, a, b):
        """Calculate the angle (in radians) between two vectors given their
        direction cosine values.

        Returns:
            theta_rad (float): angle in radians.

        """
        a = a / be.linalg.norm(a)
        b = b / be.linalg.norm(b)
        theta = be.arccos(be.clip(be.dot(a, b), -1, 1))

        return theta

    def f_number(self, n, theta):
        """Calculates the F#

        Returns:
            N_w (float): Physical F number.

        """
        N_w = 1 / (2 * n * be.sin(theta))
        return N_w

    def airy_radius(self, n_w, wavelength):
        """Calculates the airy radius
        Returns:
            r (float): Airy radius.
        """
        r = 1.22 * n_w * wavelength
        return r

    def generate_marginal_rays(self, H_x, H_y, wavelength):
        """Generates marginal rays at the stop at each pupil max,
        Px = (-1, +1), Py = (-1, +1)

        Returns:
            ray_tuple (tuple): Contains marginal each rays generated by
                trace_generic method, in north (Py=1), south (Py=-1),
                east (Px=1), west (Px=-1)directions.

        """
        ray_north = self.optic.trace_generic(
            Hx=H_x,
            Hy=H_y,
            Px=0,
            Py=1,
            wavelength=wavelength,
        )
        ray_south = self.optic.trace_generic(
            Hx=H_x,
            Hy=H_y,
            Px=0,
            Py=-1,
            wavelength=wavelength,
        )
        ray_east = self.optic.trace_generic(
            Hx=H_x,
            Hy=H_y,
            Px=1,
            Py=0,
            wavelength=wavelength,
        )
        ray_west = self.optic.trace_generic(
            Hx=H_x,
            Hy=H_y,
            Px=-1,
            Py=0,
            wavelength=wavelength,
        )

        ray_tuple = ray_north, ray_south, ray_east, ray_west

        return ray_tuple

    def generate_marginal_rays_cosines(self, H_x, H_y, wavelength):
        """Generates directional cosines for each marginal ray. Calculates one
        field at a time (one height at a time).

        Returns:
            cosines_tuple (tuple): Contains directional cosines of marginal
                each rays.

        """
        ray_north, ray_south, ray_east, ray_west = self.generate_marginal_rays(
            H_x,
            H_y,
            wavelength,
        )

        north_cosines = be.array([ray_north.L, ray_north.M, ray_north.N]).ravel()
        south_cosines = be.array([ray_south.L, ray_south.M, ray_south.N]).ravel()
        east_cosines = be.array([ray_east.L, ray_east.M, ray_east.N]).ravel()
        west_cosines = be.array([ray_west.L, ray_west.M, ray_west.N]).ravel()

        cosines_tuple = (north_cosines, south_cosines, east_cosines, west_cosines)
        return cosines_tuple

    def generate_chief_rays_cosines(self, wavelength):
        """Generates directional cosines for all chief rays of each field.

        Returns:
            chief_ray_cosines_list (ndarray): Mx3 numpy array, containing
                M number of directional cosines for all axes (x, y ,z).
            [
            field 1 -> (ray_chief1.L, ray_chief1.M, ray_chief1.N),
            field 2 -> (ray_chief1.L, ray_chief1.M, ray_chief1.N),
                .               .
                .               .
                .               .
            field M -> (ray_chiefM.L, ray_chiefM.M, ray_chiefM.N)
            ]

        """
        coords = self.fields
        chief_ray_cosines_list = []
        for H_x, H_y in coords:
            # Always pass the field values—even for 'angle' type.
            ray_chief = self.optic.trace_generic(
                Hx=H_x,
                Hy=H_y,
                Px=0,
                Py=0,
                wavelength=wavelength,
            )
            chief_ray_cosines_list.append(
                be.array([ray_chief.L, ray_chief.M, ray_chief.N]).ravel(),
            )
        chief_ray_cosines_list = be.stack(chief_ray_cosines_list, axis=0)
        return chief_ray_cosines_list

    def generate_chief_rays_centers(self, wavelength):
        """Generates the position of each chief ray. It is used to find
        centers of the airy discs in the function _plot_field.

        Returns:
            chief_ray_centers (ndarray): Contains the x, y coordinates of
                chief ray centers for each field.

        """
        coords = self.fields
        chief_ray_centers = []
        for H_x, H_y in coords:
            # Always pass the field values—even for 'angle' type.
            ray_chief = self.optic.trace_generic(
                Hx=H_x,
                Hy=H_y,
                Px=0,
                Py=0,
                wavelength=wavelength,
            )
            x, y = ray_chief.x, ray_chief.y
            chief_ray_centers.append([x, y])

        chief_ray_centers = be.stack(chief_ray_centers, axis=0)
        return chief_ray_centers

    def airy_disc_x_y(self, wavelength):
        """Generates the airy radius for each x-y axes, for each field.

        Returns:
            airy_rad_tuple (tuple): Contains x axis centers (ndarray), and y
                axis centers (ndarray).

        """
        chief_ray_cosines_list = self.generate_chief_rays_cosines(wavelength)

        airy_rad_x_list = []
        airy_rad_y_list = []

        for idx, (H_x, H_y) in enumerate(self.fields):
            # Get marginal rays for the current field
            north_cos, south_cos, east_cos, west_cos = (
                self.generate_marginal_rays_cosines(H_x, H_y, wavelength)
            )

            chief_cos = chief_ray_cosines_list[idx]

            # Compute relative angles along x and y axes
            rel_angle_north = self.angle_from_cosine(chief_cos, north_cos)
            rel_angle_south = self.angle_from_cosine(chief_cos, south_cos)
            rel_angle_east = self.angle_from_cosine(chief_cos, east_cos)
            rel_angle_west = self.angle_from_cosine(chief_cos, west_cos)

            avg_angle_x = (rel_angle_north + rel_angle_south) / 2
            avg_angle_y = (rel_angle_east + rel_angle_west) / 2

            N_w_x = self.f_number(n=1, theta=avg_angle_x)
            N_w_y = self.f_number(n=1, theta=avg_angle_y)

            airy_rad_x = self.airy_radius(N_w_x, wavelength)
            airy_rad_y = self.airy_radius(N_w_y, wavelength)

            # convert to um
            airy_rad_x_list.append(airy_rad_x * 1e-3)
            airy_rad_y_list.append(airy_rad_y * 1e-3)

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
            spot_data_item = field_data[norm_index]
            centroid_x = be.mean(spot_data_item.x)
            centroid_y = be.mean(spot_data_item.y)
            centroid.append((centroid_x, centroid_y))
        return centroid

    def geometric_spot_radius(self):
        """Geometric spot radius of each spot

        Returns:
            geometric_size (List): Geometric spot radius for field and
                wavelength

        """
        data = self._center_spots(self.data)
        geometric_size = []
        for field_data in data:
            geometric_size_field = []
            for wave_data in field_data:
                r = be.sqrt(wave_data.x**2 + wave_data.y**2)
                geometric_size_field.append(be.max(r))
            geometric_size.append(geometric_size_field)
        return geometric_size

    def rms_spot_radius(self):
        """Root mean square (RMS) spot radius of each spot

        Returns:
            rms (List): RMS spot radius for each field and wavelength.

        """
        data = self._center_spots(self.data)
        rms = []
        for field_data in data:
            rms_field = []
            for wave_data in field_data:
                r2 = wave_data.x**2 + wave_data.y**2
                rms_field.append(be.sqrt(be.mean(r2)))
            rms.append(rms_field)
        return rms

    def _center_spots(self, data):
        """Centers the spots in the given data around their respective centroids.

        Args:
            data (List): A nested list representing the data containing spots.

        Returns:
            data (List): A nested list with the spots centered around their
                centroids.

        """
        centroids = self.centroid()

        # Build a true deep copy of the nested list, cloning each array via the backend
        centered = []
        for i, field_data_list in enumerate(data):
            field_copy_list = []
            for spot_data_item in field_data_list:
                # clone each array/tensor
                x2 = be.copy(spot_data_item.x)
                y2 = be.copy(spot_data_item.y)
                i2 = be.copy(spot_data_item.intensity)

                # subtract centroid - not in-place, to prevent breaking autograd
                x2 = x2 - centroids[i][0]
                y2 = y2 - centroids[i][1]

                field_copy_list.append(SpotData(x=x2, y=y2, intensity=i2))
            centered.append(field_copy_list)
        return centered

    def _generate_data(self):
        """Generate spot data for the fields and wavelengths stored in self.

        Returns:
            data (List): A nested list of spot intersection data for each
                field and wavelength.
        """
        data = []
        # Access attributes from self
        for field in self.fields:
            field_data = []
            for (
                wavelength
            ) in self.wavelengths:  # self.wavelengths is set by BaseAnalysis
                field_data.append(
                    self._generate_field_data(
                        field,
                        wavelength,
                        self.num_rings,  # Use self.num_rings
                        self.distribution,  # Use self.distribution
                        self.coordinates,  # Use self.coordinates
                    ),
                )
            data.append(field_data)
        return data

    def _generate_field_data(
        self,
        field,
        wavelength,
        num_rays,
        distribution,
        coordinates,
    ):
        """Generates spot data for a given field and wavelength.

        Args:
            field (tuple): Tuple containing the field coordinates in (x, y).
            wavelength (float): The wavelength of the field.
            num_rays (int, optional): The number of rays to generate.
                Defaults to 100.
            distribution (str, optional): The distribution pattern of the
                rays. Defaults to 'hexapolar'.
            coordinates (str): The coordinate system ('local' or 'global').

        Returns:
            SpotData: An object containing x, y, and intensity values
                of the generated spot data.

        """
        self.optic.trace(*field, wavelength, num_rays, distribution)

        # Extract the global intersection coordinates from the image
        # surface (i.e. final surface)

        x_global = self.optic.surface_group.x[-1, :]
        y_global = self.optic.surface_group.y[-1, :]
        z_global = self.optic.surface_group.z[-1, :]
        intensity = self.optic.surface_group.intensity[-1, :]

        if coordinates == "local":
            # Now, convert the global coordinates to the image's local
            # coordinate system.
            # Do the transformation
            plot_x, plot_y, _ = transform(
                x_global, y_global, z_global, self.optic.image_surface, is_global=True
            )
        else:  # coordinates == "global"
            plot_x = x_global
            plot_y = y_global

        return SpotData(x=plot_x, y=plot_y, intensity=intensity)

    def _plot_field(
        self,
        ax,
        field_data,
        field,
        axis_lim,
        wavelengths,
        buffer=1.05,
        add_airy_disk=False,
        field_index=None,
        airy_rad_x=None,
        airy_rad_y=None,
        real_centroid_x=None,
        real_centroid_y=None,
    ):
        """Plot the field data on the given axis.

        Args:
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

        import numpy as np

        markers = ["o", "s", "^"]
        for k, points in enumerate(field_data):
            # x, y, intensity = points
            x = points.x
            y = points.y
            intensity = points.intensity
            # one‐liner conversion for BOTH backends:
            x_np = be.to_numpy(x)
            y_np = be.to_numpy(y)
            i_np = be.to_numpy(intensity)
            mask = i_np != 0
            ax.scatter(
                x_np[mask],
                y_np[mask],
                s=10,
                label=f"{wavelengths[k]:.4f} µm",
                marker=markers[k % 3],
                alpha=0.7,
            )

        if add_airy_disk and field_index is not None:
            real_centroid_x_np = be.to_numpy(real_centroid_x)
            real_centroid_y_np = be.to_numpy(real_centroid_y)
            airy_rad_x_np = be.to_numpy(airy_rad_x)
            airy_rad_y_np = be.to_numpy(airy_rad_y)
            # Draw ellipse ONLY for the current field_index
            ellipse = patches.Ellipse(
                (real_centroid_x_np, real_centroid_y_np),
                width=2 * airy_rad_y_np[field_index],  # diameter, not radius
                height=2 * airy_rad_x_np[field_index],
                linestyle="--",
                edgecolor="black",
                fill=False,
                linewidth=2,
            )
            ax.add_patch(ellipse)

            offset = abs(max(real_centroid_x_np, real_centroid_x_np))

            # Find the maximum extent among the geometric spot radius and the
            # airy disk radii.
            max_airy_x = max(airy_rad_x_np)
            max_airy_y = max(airy_rad_y_np)
            max_extent = max(axis_lim, max_airy_x, max_airy_y)

            # Apply a buffer to ensure the data fits nicely in the plot.
            adjusted_axis_lim = (offset + max_extent) * buffer
        else:
            # Without the airy disk, just use the geometric spot radius
            # with a buffer.
            adjusted_axis_lim = axis_lim * buffer

        # Determining the labels for the x and y axes based on the image
        # surface effective orientation.
        cs = self.optic.image_surface.geometry.cs
        effective_orientation = np.abs(cs.get_effective_rotation_euler())
        # Define a small tolerance to apply the new label
        tol = 0.01  # adjust it, if necessary
        if effective_orientation[0] > tol or effective_orientation[1] > tol:
            x_label, y_label = "U (mm)", "V (mm)"
        else:
            x_label, y_label = "X (mm)", "Y (mm)"

        ax.axis("square")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim((-adjusted_axis_lim, adjusted_axis_lim))
        ax.set_ylim((-adjusted_axis_lim, adjusted_axis_lim))
        ax.set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")
        ax.grid(alpha=0.25)
