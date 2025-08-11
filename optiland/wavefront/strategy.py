"""Defines the strategies for wavefront analysis.

This module provides different strategies for calculating the wavefront OPD,
each encapsulating a different algorithm for determining the reference sphere.
This approach uses the Strategy design pattern to allow for easy switching
between methods like 'chief_ray' and 'best_fit'.

Kramer Harrison, 2024
"""

from abc import ABC, abstractmethod

import optiland.backend as be

from .wavefront_data import WavefrontData


class ReferenceStrategy(ABC):
    """Abstract base class for a wavefront calculation strategy.

    This class defines the common interface for all wavefront calculation
    strategies. It also provides shared utility methods that concrete
    strategies can use, such as calculating the OPD from the image to the
    exit pupil reference sphere.

    Args:
        optic (Optic): The optical system to analyze.
        distribution (Distribution): The pupil sampling distribution.
    """

    def __init__(self, optic, distribution):
        self.optic = optic
        self.distribution = distribution
        self.n_image = optic.n()[-1]

    @abstractmethod
    def compute_wavefront_data(self, field, wavelength):
        """Orchestrates the full wavefront data calculation.

        This method should perform all necessary steps, including ray tracing,
        reference sphere calculation, and OPD computation, to generate the
        final wavefront data for a given field and wavelength.

        Args:
            field (tuple[float, float]): The field coordinates to analyze.
            wavelength (float): The wavelength to use for the analysis.

        Returns:
            WavefrontData: A data object containing the results.
        """
        pass

    def _opd_image_to_xp(self, rays_at_image, xc, yc, zc, R, wavelength):
        """Computes propagation distance from image plane to exit pupil sphere.

        Args:
            rays_at_image (object): An object with ray data attributes (x, y, z,
                L, M, N) at the final image surface.
            xc (float): The x-coordinate of the reference sphere center.
            yc (float): The y-coordinate of the reference sphere center.
            zc (float): The z-coordinate of the reference sphere center.
            R (float): The radius of the reference sphere.
            wavelength (float): The wavelength of the light.

        Returns:
            float: The optical path length from the image surface to the
                   reference sphere.
        """
        xr, yr, zr = rays_at_image.x, rays_at_image.y, rays_at_image.z
        L, M, N = -rays_at_image.L, -rays_at_image.M, -rays_at_image.N

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
        d = be.where(d < 0, 0, d)  # Ensure non-negative for sqrt
        t = (-b - be.sqrt(d)) / (2 * a)

        # If the first solution for t is negative, the ray is pointing away
        # from the sphere, so we take the other root.
        mask = t < 0
        t = be.where(mask, (-b + be.sqrt(d)) / (2 * a), t)

        return self.n_image * t

    def _correct_tilt(self, field, opd, x=None, y=None):
        """Corrects for tilt in the OPD based on the field angle.

        Args:
            field (tuple[float, float]): The field coordinates (Hx, Hy).
            opd (ndarray): The optical path difference array to correct.
            x (ndarray, optional): The x-coordinates of the pupil distribution.
                If None, uses the strategy's distribution. Defaults to None.
            y (ndarray, optional): The y-coordinates of the pupil distribution.
                If None, uses the strategy's distribution. Defaults to None.

        Returns:
            ndarray: The OPD array with tilt correction applied.
        """
        correction = 0
        if self.optic.field_type == "angle":
            hx, hy = field
            max_f = self.optic.fields.max_field
            x_tilt = max_f * hx
            y_tilt = max_f * hy
            xs = self.distribution.x if x is None else x
            ys = self.distribution.y if y is None else y
            epd = self.optic.paraxial.EPD()
            correction = (1 - xs) * be.sin(be.radians(x_tilt)) * epd / 2 + (
                1 - ys
            ) * be.sin(be.radians(y_tilt)) * epd / 2
        return opd - correction


class ChiefRayStrategy(ReferenceStrategy):
    """Calculates wavefront using the chief ray as the reference."""

    def __init__(self, optic, distribution):
        super().__init__(optic, distribution)
        self.pupil_z = optic.paraxial.XPL() + optic.surface_group.positions[-1]

    def compute_wavefront_data(self, field, wavelength):
        """Computes wavefront data using the chief ray reference method.

        This method preserves the original calculation logic:
        1. Trace the chief ray to define the reference sphere.
        2. Calculate the reference OPD from this chief ray.
        3. Trace the full set of rays for the field.
        4. Compute the OPD for all rays relative to the reference sphere.
        5. Normalize the final OPD map using the chief ray's reference OPD.

        Args:
            field (tuple[float, float]): The field coordinates to analyze.
            wavelength (float): The wavelength to use for the analysis.

        Returns:
            WavefrontData: A data object containing the results.
        """
        # 1. Trace chief ray and determine reference sphere
        chief_ray = self.optic.trace_generic(
            *field, Px=0.0, Py=0.0, wavelength=wavelength
        )
        xc, yc, zc, R = self._calculate_sphere_from_chief_ray(chief_ray)

        # 2. Calculate reference OPD from the chief ray
        opd_img_ref = self._opd_image_to_xp(chief_ray, xc, yc, zc, R, wavelength)
        opd_ref = chief_ray.opd - opd_img_ref
        opd_ref = self._correct_tilt(field, opd_ref, x=0, y=0)

        # 3. Trace the full grid of rays for the field
        rays = self.optic.trace(*field, wavelength, None, self.distribution)
        intensity = self.optic.surface_group.intensity[-1, :]

        # 4. Compute OPD for all rays
        opd_img = self._opd_image_to_xp(rays, xc, yc, zc, R, wavelength)
        opd = rays.opd - opd_img
        opd = self._correct_tilt(field, opd)

        # 5. Normalize OPD and calculate pupil coordinates
        opd_wv = (opd_ref - opd) / (wavelength * 1e-3)
        t = opd_img / self.n_image
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

    def _calculate_sphere_from_chief_ray(self, chief_ray):
        """Determine reference sphere center and radius from a chief ray."""
        x, y, z = chief_ray.x, chief_ray.y, chief_ray.z
        if be.size(x) != 1:
            raise ValueError("Chief ray cannot be determined. It must be traced alone.")
        R = be.sqrt(x**2 + y**2 + (z - self.pupil_z) ** 2)
        return x, y, z, R


class BestFitStrategy(ReferenceStrategy):
    """Calculates wavefront using a best-fit reference sphere.

    Fits the sphere in two stages:
    1. Algebraic least squares for initial estimate.
    2. Optional Gauss-Newton iterative refinement for improved accuracy.

    Scaling is applied for numerical robustness.
    """

    def __init__(
        self,
        optic,
        distribution,
        max_iter: int = 20,
        tol: float = 1e-6,
        refine_fit: bool = True,
    ):
        """Initialize the best-fit strategy.

        Args:
            max_iter (int, optional): Max iterations for Gauss-Newton refinement.
            tol (float, optional): Convergence tolerance for parameter updates.
            refine_fit (bool, optional): Whether to perform Gauss-Newton refinement.
        """
        super().__init__(optic, distribution)
        self.max_iter = max_iter
        self.tol = tol
        self.refine_fit = refine_fit

    def compute_wavefront_data(self, field, wavelength):
        """Compute wavefront data using the best-fit reference sphere method.

        Steps:
            1. Trace the full set of rays for the field.
            2. Fit a sphere to the wavefront points.
            3. Compute the OPD relative to this sphere.
            4. Use the minimum OPD as the reference OPD.
            5. Normalize the OPD map.

        Args:
            field (tuple[float, float]): Field coordinates to analyze.
            wavelength (float): Wavelength for the analysis.

        Returns:
            WavefrontData: Computed wavefront data.
        """
        rays = self.optic.trace(*field, wavelength, None, self.distribution)
        intensity = self.optic.surface_group.intensity[-1, :]

        xc, yc, zc, R = self._calculate_best_fit_sphere(rays)

        opd_img = self._opd_image_to_xp(rays, xc, yc, zc, R, wavelength)
        opd = rays.opd - opd_img
        opd_ref = be.min(opd)

        opd_wv = (opd_ref - opd) / (wavelength * 1e-3)
        t = opd_img / self.n_image
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

    def _calculate_best_fit_sphere(self, rays):
        """Fit a sphere to the wavefront points."""
        pts, _ = self._points_from_rays(rays)
        center, radius = self._fit_sphere(pts)
        return *center, radius

    def _points_from_rays(self, rays):
        """Convert ray data to 3D wavefront points."""
        valid = (
            be.isfinite(rays.x)
            & be.isfinite(rays.y)
            & be.isfinite(rays.z)
            & be.isfinite(rays.L)
            & be.isfinite(rays.M)
            & be.isfinite(rays.N)
            & be.isfinite(rays.opd)
            & (rays.i != 0)
        )
        if not be.any(valid):
            raise ValueError("No valid ray samples found for best-fit sphere.")

        p = be.stack((rays.x, rays.y, rays.z), axis=1)[valid]
        d = be.stack((rays.L, rays.M, rays.N), axis=1)[valid]
        s = rays.opd[valid] / self.n_image
        pts = p - s[:, None] * d
        return pts, valid

    def _fit_sphere(self, pts):
        """Fit a sphere using algebraic LS and optional refinement."""
        q, mean, scale = self._scale_points(pts)
        theta = self._algebraic_fit(q)
        if self.refine_fit:
            theta = self._gauss_newton_refine(q, theta)
        center = scale * theta[:3] + mean
        radius = float(scale * theta[3])
        return tuple(center), radius

    def _scale_points(self, pts):
        """Scale points for numerical stability.

        Returns:
            tuple: (scaled_points, mean, scale_factor)
        """
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("Sphere fit points must be shape (N, 3).")

        mean = pts.mean(axis=0)
        scale = be.ptp(pts, axis=0).max() or 1.0
        q = (pts - mean) / scale
        return q, mean, scale

    def _algebraic_fit(self, q):
        """Algebraic least squares sphere fit.

        Args:
            q (array): Scaled points (N, 3).

        Returns:
            array: Initial parameters [cx, cy, cz, r].
        """
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        A = be.column_stack((x, y, z, be.ones_like(x)))
        b = x**2 + y**2 + z**2
        D, E, F, G = be.linalg.lstsq(A, b, rcond=None)[0]
        cx, cy, cz = -D / 2, -E / 2, -F / 2
        r = be.sqrt(cx**2 + cy**2 + cz**2 + G)
        return be.array([cx, cy, cz, r])

    def _gauss_newton_refine(self, q, theta):
        """Refine sphere parameters with Gauss-Newton iterations.

        Args:
            q (array): Scaled points (N, 3).
            theta (array): Initial parameters [cx, cy, cz, r].

        Returns:
            array: Refined parameters [cx, cy, cz, r].
        """
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        points = be.column_stack((x, y, z))

        for _ in range(self.max_iter):
            diffs = points - theta[:3]
            dists = be.linalg.norm(diffs, axis=1)
            residuals = dists - theta[3]

            J = be.empty((len(points), 4))
            J[:, 0] = (theta[0] - x) / dists
            J[:, 1] = (theta[1] - y) / dists
            J[:, 2] = (theta[2] - z) / dists
            J[:, 3] = -1.0

            delta = be.linalg.solve(J.T @ J, -J.T @ residuals)
            theta += delta

            if be.linalg.norm(delta) < self.tol:
                break

        return theta


def create_strategy(strategy_name, optic, distribution, **kwargs):
    """Factory function to create a wavefront calculation strategy.

    Args:
        strategy_name (str): The name of the strategy ("chief_ray", "best_fit").
        optic (Optic): The optical system.
        distribution (Distribution): The pupil sampling distribution.

    Returns:
        ReferenceStrategy: An instance of the requested strategy.

    Raises:
        ValueError: If the strategy_name is unknown.
    """
    strategies = {
        "chief_ray": ChiefRayStrategy,
        "best_fit": BestFitStrategy,
    }
    strategy_class = strategies.get(strategy_name)

    if strategy_class:
        return strategy_class(optic, distribution, **kwargs)
    else:
        raise ValueError(f"Unknown wavefront strategy: {strategy_name}")
