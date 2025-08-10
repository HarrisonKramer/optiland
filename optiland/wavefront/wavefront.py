"""
This module defines the `Wavefront` class, which is designed to analyze the
wavefront of an optical system.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.distribution import create_distribution
from optiland.wavefront.reference_sphere import create_reference_sphere_calculator
from optiland.wavefront.wavefront_data import WavefrontData


class Wavefront:
    """
    Performs wavefront analysis on an optical system.

    Computes ray intersection points with the exit pupil, the optical path
    difference (OPD) relative to a reference sphere, the radius of curvature
    of the exit-pupil reference sphere, and ray intensities.

    Args:
        optic (Optic): The optical system to analyze.
        fields (str or List[Tuple[float, float]]): Fields or 'all'.
        wavelengths (str or List[float]): Wavelengths or 'all'/'primary'.
        num_rays (int): Number of rays for pupil sampling.
        distribution (str or Distribution): Ray distribution or its name.
        ref_sphere_calculator (str): The reference sphere calculation strategy.
            Defaults to 'chief_ray'.

    Attributes:
        data (List[List[WavefrontData]]): Nested lists indexed
            by [field][wavelength].
    """

    def __init__(
        self,
        optic,
        fields="all",
        wavelengths="all",
        num_rays=12,
        distribution="hexapolar",
        ref_sphere_calculator="chief_ray",
    ):
        self.optic = optic
        self.fields = self._resolve_fields(fields)
        self.wavelengths = self._resolve_wavelengths(wavelengths)
        self.num_rays = num_rays
        self.distribution = self._resolve_distribution(distribution, self.num_rays)
        self.ref_sphere_calculator = create_reference_sphere_calculator(
            ref_sphere_calculator, self.optic
        )
        self.data = {}
        self._generate_data()

    def get_data(self, field, wl):
        """
        Retrieve precomputed wavefront data for a given field and wavelength.

        Args:
            field (Tuple[float, float]): Field coordinates.
            wl (float): Wavelength.

        Returns:
            WavefrontData: Data container with intersections, OPD, intensity, and
            curvature.
        """
        return self.data[(field, wl)]

    def _resolve_fields(self, fields):
        """Resolve field coordinates based on input."""
        if fields == "all":
            return self.optic.fields.get_field_coords()
        return fields

    def _resolve_wavelengths(self, wavelengths):
        """Resolve wavelengths based on input."""
        if wavelengths == "all":
            return self.optic.wavelengths.get_wavelengths()
        if wavelengths == "primary":
            return [self.optic.primary_wavelength]
        return wavelengths

    def _resolve_distribution(self, dist, num_rays):
        """Resolve distribution based on input."""
        if isinstance(dist, str):
            dist_obj = create_distribution(dist)
            dist_obj.generate_points(num_rays)
            return dist_obj
        return dist

    def _generate_data(self):
        """Generate wavefront data for all fields and wavelengths."""
        for field in self.fields:
            for wl in self.wavelengths:
                # trace chief ray and get reference sphere
                self._trace_chief_ray(field, wl)
                xc, yc, zc, R = self.ref_sphere_calculator.calculate()

                # reference OPD (chief ray)
                opd_ref, _ = self._get_path_length(xc, yc, zc, R, wl)
                opd_ref = self._correct_tilt(field, opd_ref, x=0, y=0)

                # generate full field data
                self.data[(field, wl)] = self._generate_field_data(
                    field, wl, opd_ref, xc, yc, zc, R
                )

    def _generate_field_data(self, field, wavelength, opd_ref, xc, yc, zc, R):
        """
        Generate WavefrontData for a single field and wavelength.

        Args:
            field (tuple): Field coordinates.
            wavelength (float): Wavelength.
            opd_ref: Reference OPD from chief ray.
            xc, yc, zc: Reference sphere center.
            R: Reference sphere radius.

        Returns:
            WavefrontData: All per-ray results.
        """
        rays = self.optic.trace(*field, wavelength, None, self.distribution)
        intensity = self.optic.surface_group.intensity[-1, :]

        opd, t = self._get_path_length(xc, yc, zc, R, wavelength)
        opd = self._correct_tilt(field, opd)
        opd_wv = (opd_ref - opd) / (wavelength * 1e-3)  # OPD map in waves

        pupil_x = rays.x - t * rays.L
        pupil_y = rays.y - t * rays.M
        pupil_z = rays.z - t * rays.N
        return WavefrontData(
            pupil_x=pupil_x,
            pupil_y=pupil_y,
            pupil_z=pupil_z,
            opd=opd_wv,
            intensity=intensity,
            radius=R,
        )

    def _trace_chief_ray(self, field, wavelength):
        """
        Trace the chief ray for a given field and wavelength.
        """
        self.optic.trace_generic(*field, Px=0.0, Py=0.0, wavelength=wavelength)

    def _get_path_length(self, xc, yc, zc, R, wavelength):
        """
        Calculate optical path difference from image to reference sphere.
        """
        opd_chief = self.optic.surface_group.opd[-1, :]
        opd_img = self._opd_image_to_xp(xc, yc, zc, R, wavelength)
        return opd_chief - opd_img, opd_img

    def _correct_tilt(self, field, opd, x=None, y=None):
        """
        Correct tilt in OPD based on field angle and distribution.
        """
        correction = 0
        if self.optic.field_type == "angle":
            Hx, Hy = field
            max_f = self.optic.fields.max_field
            x_tilt = max_f * Hx
            y_tilt = max_f * Hy
            xs = self.distribution.x if x is None else x
            ys = self.distribution.y if y is None else y
            EPD = self.optic.paraxial.EPD()
            correction = (1 - xs) * be.sin(be.radians(x_tilt)) * EPD / 2 + (
                1 - ys
            ) * be.sin(be.radians(y_tilt)) * EPD / 2
        return opd - correction

    def _opd_image_to_xp(self, xc, yc, zc, R, wavelength):
        """
        Compute propagation distance from image plane to exit pupil.
        """
        xr = self.optic.surface_group.x[-1, :]
        yr = self.optic.surface_group.y[-1, :]
        zr = self.optic.surface_group.z[-1, :]
        L = -self.optic.surface_group.L[-1, :]
        M = -self.optic.surface_group.M[-1, :]
        N = -self.optic.surface_group.N[-1, :]
        a = L**2 + M**2 + N**2
        b = 2 * (L * (xr - xc) + M * (yr - yc) + N * (zr - zc))
        c = (
            xr**2
            + yr**2
            + zr**2
            - 2 * (xr * xc + yr * yc + zr * zc)
            + xc**2
            + yc**2
            + zc**2
            - R**2
        )
        d = b**2 - 4 * a * c
        t = (-b - be.sqrt(d)) / (2 * a)
        mask = t < 0
        t = be.where(mask, (-b + be.sqrt(d)) / (2 * a), t)
        n = self.optic.image_surface.material_post.n(wavelength)
        return n * t
