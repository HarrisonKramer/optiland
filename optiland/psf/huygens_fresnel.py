"""Huygens-Fresnel Point Spread Function (PSF) Module

This module provides functionality for simulating and analyzing the Point
Spread Function (PSF) of optical systems using the Huygens-Fresnel principle.
It includes capabilities for generating PSF from given wavefront aberrations.
Visualization and Strehl ratio calculation are primarily handled by the base
class. The PSF is normalized against the peak of an ideal diffraction-limited
system calculated using the same Huygens-Fresnel principle.

Kramer Harrison, 2025
"""

import optiland.backend as be
from optiland.psf.base import BasePSF
from optiland.rays import RealRays


class HuygensPSF(BasePSF):
    """Huygens PSF class using Huygens-Fresnel principle.

    Computes PSF by integrating contributions from point sources across the
    exit pupil. The PSF is normalized such that the peak intensity of an
    equivalent diffraction-limited system (same pupil amplitude, zero phase error)
    would be 100.0. This makes the Strehl ratio (peak of actual PSF / 100.0,
    or value at PSF center / 100.0) meaningful.

    Inherits from `BasePSF` for common initialization (Wavefront setup) and
    visualization methods.
    """

    def __init__(
        self, optic, field, wavelength, num_rays=128, image_size=128, grid_size=1024
    ):
        """Initializes the HuygensPSF object."""
        super().__init__(
            optic=optic,
            field=field,
            wavelength=wavelength,
            num_rays=num_rays,
            grid_size=grid_size,
        )
        self.image_size = image_size
        self.psf = self._compute_psf()

    def _get_pupil_data(self, Hx, Hy, wavelength_um, zero_phase: bool = False):
        """Helper to get pupil data in global coordinates.

        Args:
            Hx (float): Normalized field coordinate X.
            Hy (float): Normalized field coordinate Y.
            wavelength_um (float): Wavelength in micrometers.
            zero_phase (bool): If True, the pupil phase (derived from OPD)
                               is set to zero, simulating a diffraction-limited
                               wavefront. The pupil amplitude (transmission)
                               remains as per the system.

        Returns:
            tuple: Contains (pupil_x_g, pupil_y_g, pupil_z_g, pupil_amplitude,
                     pupil_phase, pupil_normal_x_g, pupil_normal_y_g, pupil_normal_z_g)
                     all as backend arrays.
        """
        data = self.get_data((Hx, Hy), wavelength_um)  # From Wavefront class

        pupil_x_g = data.pupil_x
        pupil_y_g = data.pupil_y
        pupil_z_g = data.pupil_z

        pupil_amplitude = be.sqrt(data.intensity)
        mean_amp = be.mean(pupil_amplitude)
        if mean_amp > 1e-9:  # Avoid division by zero
            pupil_amplitude = pupil_amplitude / mean_amp
        else:
            pupil_amplitude = be.zeros_like(pupil_amplitude)

        if zero_phase:
            # For diffraction-limited calculation, phase error is zero.
            pupil_phase = be.zeros_like(pupil_x_g, dtype=be.float_dtype())
        else:
            # For actual PSF, use phase from OPD.
            k_um = 2.0 * be.pi / wavelength_um
            pupil_phase = k_um * data.opd  # OPD is path diff in um

        # Pupil normals calculation
        # A more robust solution would be for Wavefront.get_data() to provide
        # global normals,
        # or to compute them based on the exit pupil sphere's center and radius.
        # This simplified version assumes pupil coordinates are relative to EPP center,
        # and that this center is at the origin of the coordinate system in which these
        # points are given.
        if (
            hasattr(data, "pupil_nx") and data.pupil_nx is not None
        ):  # Prefer direct normals if available
            pupil_normal_x_g = data.pupil_nx
            pupil_normal_y_g = data.pupil_ny
            pupil_normal_z_g = data.pupil_nz
        else:  # Fallback simplified normal calculation
            Rp_pupil = data.radius  # Radius of the exit pupil sphere
            if Rp_pupil == 0:
                Rp_pupil = 1e-9  # avoid div by zero

            temp_norm_x = pupil_x_g / Rp_pupil
            temp_norm_y = pupil_y_g / Rp_pupil
            temp_norm_z = pupil_z_g / Rp_pupil

            norm_lengths = be.sqrt(temp_norm_x**2 + temp_norm_y**2 + temp_norm_z**2)
            norm_lengths = be.where(
                norm_lengths == 0, 1.0, norm_lengths
            )  # Avoid div by zero

            pupil_normal_x_g = temp_norm_x / norm_lengths
            pupil_normal_y_g = temp_norm_y / norm_lengths
            pupil_normal_z_g = temp_norm_z / norm_lengths

        return (
            pupil_x_g,
            pupil_y_g,
            pupil_z_g,
            pupil_amplitude,
            pupil_phase,
            pupil_normal_x_g,
            pupil_normal_y_g,
            pupil_normal_z_g,
        )

    def _get_image_plane_coords(self, Hx, Hy, wavelength_um):
        """Helper to get image plane coordinates in the global system.
        The grid is centered around the chief ray (or centroid) intersection
        on the image surface.
        """
        N = self.image_size
        lens = self.optic
        image_surface = lens.image_surface
        cs_image = image_surface.geometry.cs

        # 1. Trace rays to find the global centroid of ray intersections on the image
        # surface.
        trace_rays_for_centroid = lens.trace(
            Hx=Hx,
            Hy=Hy,
            wavelength=wavelength_um,
            distribution="hexapolar",
            num_rays=12,
        )
        cx_g_centroid = be.mean(trace_rays_for_centroid.x)
        cy_g_centroid = be.mean(trace_rays_for_centroid.y)
        cz_g_centroid = be.mean(trace_rays_for_centroid.z)

        # Create a RealRays object for the global centroid point to transform it
        # Convert to numpy then to backend array to ensure correct scalar-to-array
        # conversion for RealRays
        P_centroid_global_rays = RealRays(
            x=be.array([be.to_numpy(cx_g_centroid).item()]),
            y=be.array([be.to_numpy(cy_g_centroid).item()]),
            z=be.array([be.to_numpy(cz_g_centroid).item()]),
            L=be.array([0.0]),
            M=be.array([0.0]),
            N=be.array([1.0]),  # Dummy directions
            opd=be.array([0.0]),
            intensity=be.array([1.0]),
            wavelength=wavelength_um,
        )

        # 2. Transform this global patch center to the local coordinates of the image
        # surface's CS.
        cs_image.localize(
            P_centroid_global_rays
        )  # Modifies P_centroid_global_rays in place
        cx_local_img_cs = P_centroid_global_rays.x[0]  # Should be scalar now
        cy_local_img_cs = P_centroid_global_rays.y[0]

        # Determine physical extent of the grid based on ray spread (global)
        extent_x = be.max(be.abs(trace_rays_for_centroid.x - cx_g_centroid))
        extent_y = be.max(be.abs(trace_rays_for_centroid.y - cy_g_centroid))
        physical_extent = be.max(be.stack([extent_x, extent_y])) + 1e-9  # Add epsilon

        # 3. Create a 2D grid of offsets, to be applied in the local XY plane of the
        # image surface CS
        half_extent = physical_extent / 2.0
        x_1d_offsets = be.linspace(-half_extent, half_extent, N)
        y_1d_offsets = be.linspace(-half_extent, half_extent, N)
        x_grid_offsets, y_grid_offsets = be.meshgrid(x_1d_offsets, y_1d_offsets)

        # 4. Define (x,y) coordinates in the local CS of the image surface for sag
        # calculation.
        img_x_local_for_sag = cx_local_img_cs + x_grid_offsets
        img_y_local_for_sag = cy_local_img_cs + y_grid_offsets

        # 5. Calculate sag values (z-coordinates in local CS) for these (x,y) points.
        img_z_local_sag_values = image_surface.geometry.sag(
            img_x_local_for_sag, img_y_local_for_sag
        )

        # Flatten for transformation
        local_points_x_flat = be.ravel(img_x_local_for_sag)
        local_points_y_flat = be.ravel(img_y_local_for_sag)
        local_points_z_flat_sag = be.ravel(img_z_local_sag_values)

        num_total_points = N * N
        # Create RealRays object with these local points to use cs_image.globalize()
        local_surface_points_rays = RealRays(
            x=local_points_x_flat,
            y=local_points_y_flat,
            z=local_points_z_flat_sag,
            L=be.zeros(num_total_points, dtype=be.float_dtype()),
            M=be.zeros(num_total_points, dtype=be.float_dtype()),
            N=be.ones(
                num_total_points, dtype=be.float_dtype()
            ),  # Pointing along local Z
            opd=be.zeros(num_total_points, dtype=be.float_dtype()),
            intensity=be.ones(num_total_points, dtype=be.float_dtype()),
            wavelength=wavelength_um,
        )

        # 6. Transform these local surface points to global coordinates.
        cs_image.globalize(local_surface_points_rays)  # Modifies in place

        # Reshape back to (N,N) grids
        img_x_be = be.reshape(local_surface_points_rays.x, (N, N))
        img_y_be = be.reshape(local_surface_points_rays.y, (N, N))
        img_z_be = be.reshape(local_surface_points_rays.z, (N, N))

        return img_x_be, img_y_be, img_z_be

    def _compute_diffraction_limited_peak_intensity(self):
        """Computes the on-axis peak intensity for an equivalent diffraction-limited
        system.

        This calculation uses the same pupil amplitude distribution as the actual
        system but assumes a perfect (zero OPD) wavefront. The intensity is
        calculated at the center of the image grid defined for an on-axis field.
        This serves as the reference for Strehl ratio calculation.
        """
        wavelength_um = self.wavelengths[0]  # Primary wavelength
        N_image = self.image_size  # Grid size for finding center point
        wavelength_mm = (
            wavelength_um * 1e-3
        )  # Convert um to mm for optical calculations

        # Get pupil data with zero phase, for on-axis field (Hx=0, Hy=0)
        # This uses the actual system's pupil amplitude distribution.
        (
            pupil_x_g,
            pupil_y_g,
            pupil_z_g,
            pupil_amp,
            ideal_pupil_phase,  # ideal_pupil_phase is zeros
            pupil_nx_g,
            pupil_ny_g,
            pupil_nz_g,
        ) = self._get_pupil_data(0.0, 0.0, wavelength_um, zero_phase=True)

        # Get image plane coordinates for on-axis field
        img_x_g_grid, img_y_g_grid, img_z_g_grid = self._get_image_plane_coords(
            0.0, 0.0, wavelength_um
        )

        center_idx = N_image // 2
        # Ensure scalar values for single point calculation by converting to NumPy
        # scalar then back to backend if needed
        ix_on_axis = be.array(be.to_numpy(img_x_g_grid[center_idx, center_idx]).item())
        iy_on_axis = be.array(be.to_numpy(img_y_g_grid[center_idx, center_idx]).item())
        iz_on_axis = be.array(be.to_numpy(img_z_g_grid[center_idx, center_idx]).item())

        # Huygens-Fresnel summation for this single on-axis image point
        # pupil_x_g etc. are 1D arrays [M_pupil_points]
        dx = ix_on_axis - pupil_x_g
        dy = iy_on_axis - pupil_y_g
        dz = iz_on_axis - pupil_z_g

        R = be.sqrt(dx**2 + dy**2 + dz**2)
        R = be.where(R == 0, 1e-9, R)  # Avoid division by zero

        k_mm = 2.0 * be.pi / wavelength_mm
        # complex_pupil_contribution uses actual pupil_amp and zero
        # phase (ideal_pupil_phase)
        complex_pupil_contribution = pupil_amp * be.exp(1j * ideal_pupil_phase)
        propagator = be.exp(1j * k_mm * R) / R

        # Obliquity factor calculation
        dot_prod_obliq = dx * pupil_nx_g + dy * pupil_ny_g + dz * pupil_nz_g
        cos_theta_prime = dot_prod_obliq / R
        obliquity_factor = 0.5 * (1.0 + cos_theta_prime)

        integrand_on_axis = complex_pupil_contribution * propagator * obliquity_factor
        summed_field_on_axis = be.sum(integrand_on_axis)  # Sum over all pupil points

        constant_factor = 1.0 / (1j * wavelength_mm)
        E_ideal_on_axis = constant_factor * summed_field_on_axis

        ideal_peak_intensity = be.real(E_ideal_on_axis * be.conj(E_ideal_on_axis))

        return ideal_peak_intensity

    def _compute_psf(self):
        """Computes the PSF, normalized against its diffraction-limited peak."""
        Hx, Hy = self.fields[0]  # Current field point from BasePSF
        wavelength_um = self.wavelengths[0]  # Primary wavelength from BasePSF
        N_image = self.image_size
        wavelength_mm = (
            wavelength_um * 1e-3
        )  # Convert um to mm for optical calculations

        # Get actual pupil data (with aberrations for the current field)
        (
            pupil_x_g,
            pupil_y_g,
            pupil_z_g,
            pupil_amp,
            pupil_phase,  # pupil_phase has aberrations
            pupil_nx_g,
            pupil_ny_g,
            pupil_nz_g,
        ) = self._get_pupil_data(Hx, Hy, wavelength_um, zero_phase=False)

        # Get image plane coordinates for the current field
        img_x_g, img_y_g, img_z_g = self._get_image_plane_coords(Hx, Hy, wavelength_um)

        # Flatten image coordinates for broadcasting
        img_x_flat = be.ravel(img_x_g)
        img_y_flat = be.ravel(img_y_g)
        img_z_flat = be.ravel(img_z_g)

        # Expand dims for broadcasting: pupil arrays (1, M_pupil_points), image
        # arrays (N_image*N_image, 1)
        pup_x_exp = be.expand_dims(pupil_x_g, 0)
        pup_y_exp = be.expand_dims(pupil_y_g, 0)
        pup_z_exp = be.expand_dims(pupil_z_g, 0)
        pup_amp_exp = be.expand_dims(pupil_amp, 0)
        pup_phase_exp = be.expand_dims(pupil_phase, 0)  # Actual phase with aberrations
        pup_nx_exp = be.expand_dims(pupil_nx_g, 0)
        pup_ny_exp = be.expand_dims(pupil_ny_g, 0)
        pup_nz_exp = be.expand_dims(pupil_nz_g, 0)

        img_x_exp = be.expand_dims(img_x_flat, 1)
        img_y_exp = be.expand_dims(img_y_flat, 1)
        img_z_exp = be.expand_dims(img_z_flat, 1)

        # Calculate field for all image points from all pupil points
        dx = img_x_exp - pup_x_exp
        dy = img_y_exp - pup_y_exp
        dz = img_z_exp - pup_z_exp

        R = be.sqrt(dx**2 + dy**2 + dz**2)
        R = be.where(R == 0, 1e-9, R)  # Avoid division by zero

        k_mm = 2.0 * be.pi / wavelength_mm
        complex_pupil_field = pup_amp_exp * be.exp(
            1j * pup_phase_exp
        )  # Use actual phase
        propagator = be.exp(1j * k_mm * R) / R

        dot_prod_obliq = dx * pup_nx_exp + dy * pup_ny_exp + dz * pup_nz_exp
        cos_theta_prime = dot_prod_obliq / R
        obliquity_factor = 0.5 * (1.0 + cos_theta_prime)

        integrand = complex_pupil_field * propagator * obliquity_factor
        summed_field_flat = be.sum(integrand, axis=1)  # Sum over pupil points (axis 1)

        constant_factor = 1.0 / (1j * wavelength_mm)
        complex_field_flat = constant_factor * summed_field_flat
        complex_field_image = be.reshape(complex_field_flat, (N_image, N_image))

        raw_psf = be.real(complex_field_image * be.conj(complex_field_image))

        # Normalize against diffraction-limited peak intensity
        ideal_peak = self._compute_diffraction_limited_peak_intensity()

        if ideal_peak > 1e-12:  # Ensure ideal_peak is not zero or extremely small
            normalized_psf = (raw_psf / ideal_peak) * 100.0
        else:
            # Fallback if ideal peak is zero (e.g. fully vignetted system)
            # Return zeros, as Strehl would be undefined or zero.
            normalized_psf = be.zeros_like(raw_psf)

        return normalized_psf

    def _get_psf_units(self, image_data_for_shape):
        """Calculates the physical extent (units) of the PSF image for plotting."""
        Hx, Hy = self.fields[0]
        wavelength_um = self.wavelengths[0]
        lens = self.optic

        trace_rays = lens.trace(
            Hx=Hx,
            Hy=Hy,
            wavelength=wavelength_um,
            distribution="hexapolar",
            num_rays=12,
        )
        cx_g = be.mean(trace_rays.x)
        cy_g = be.mean(trace_rays.y)
        extent_x = be.max(be.abs(trace_rays.x - cx_g))
        extent_y = be.max(be.abs(trace_rays.y - cy_g))
        physical_extent_mm = be.max(be.stack([extent_x, extent_y])) + 1e-9

        # self.image_size is N, the number of points in one dimension of the image grid
        pixel_size_mm = physical_extent_mm / self.image_size

        # image_data_for_shape is the PSF array, its shape is (N, N) in this context
        total_width_mm = image_data_for_shape.shape[1] * pixel_size_mm
        total_height_mm = image_data_for_shape.shape[0] * pixel_size_mm

        total_width_um = total_width_mm * 1000.0
        total_height_um = total_height_mm * 1000.0

        return be.to_numpy(total_width_um), be.to_numpy(total_height_um)
