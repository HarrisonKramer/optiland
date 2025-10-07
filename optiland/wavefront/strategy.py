"""Defines the strategies for wavefront analysis.

This module provides different strategies for calculating the wavefront OPD,
each encapsulating a different algorithm for determining the reference sphere.
This approach uses the Strategy design pattern to allow for easy switching
between methods like 'chief_ray' and 'centroid_sphere'.

Kramer Harrison, 2024
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import optiland.backend as be

from ..fields.field_types import AngleField
from .wavefront_data import WavefrontData

if TYPE_CHECKING:
    from optiland._types import BEArrayT
    from optiland.distribution import BaseDistribution
    from optiland.optic.optic import Optic
    from optiland.rays.base import BaseRays
    from optiland.rays.real_rays import RealRays

WavefrontStrategyType = Literal["chief_ray", "centroid_sphere", "best_fit_sphere"]


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

    def __init__(self, optic: Optic, distribution: BaseDistribution, **kwargs) -> None:
        self.optic = optic
        self.distribution = distribution
        self.n_image = optic.n()[-1]

    @abstractmethod
    def compute_wavefront_data(
        self, field: tuple[float, float], wavelength: float
    ) -> WavefrontData:
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

    def _opd_image_to_xp(
        self,
        rays_at_image: RealRays,
        xc: float,
        yc: float,
        zc: float,
        R: float,
        wavelength: float,
    ) -> float:
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

    def _correct_tilt(
        self,
        field: tuple[float, float],
        opd: BEArrayT,
        x: BEArrayT | float | None = None,
        y: BEArrayT | float | None = None,
    ) -> BEArrayT:
        """Corrects for tilt in the OPD based on the field angle.

        This step is needed because, in the case of angular fields, rays launch from a
        plane at z=const in object space. This results in an artificla tilt of the
        wavefront that must be removed prior to wavefront calculations.

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
        if not isinstance(self.optic.field_definition, AngleField):
            return opd

        hx, hy = field
        max_field_deg = self.optic.fields.max_field
        fx = hx * max_field_deg
        fy = hy * max_field_deg
        fx_rad = be.deg2rad(fx)
        fy_rad = be.deg2rad(fy)

        # direction cosines
        tx, ty = be.tan(fx_rad), be.tan(fy_rad)
        uz = 1.0 / be.sqrt(1.0 + tx**2 + ty**2)
        ux, uy = tx * uz, ty * uz

        # physical pupil coords
        xs = be.array(self.distribution.x) if x is None else be.array(x)
        ys = be.array(self.distribution.y) if y is None else be.array(y)
        epd = self.optic.paraxial.EPD()
        X_m = xs * epd / 2
        Y_m = ys * epd / 2

        # remove artificial tilt from launch plane
        tilt = ux * X_m + uy * Y_m
        return opd + tilt


class ChiefRayStrategy(ReferenceStrategy):
    """Calculates wavefront using the chief ray as the reference."""

    def __init__(self, optic: Optic, distribution: BaseDistribution, **kwargs) -> None:
        super().__init__(optic, distribution, **kwargs)
        self.pupil_z = optic.paraxial.XPL() + optic.surface_group.positions[-1]

    def compute_wavefront_data(
        self, field: tuple[float, float], wavelength: float
    ) -> WavefrontData:
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

    def _calculate_sphere_from_chief_ray(
        self, chief_ray: BaseRays
    ) -> tuple[float, float, float, float]:
        """Determine reference sphere center and radius from a chief ray."""
        x, y, z = chief_ray.x, chief_ray.y, chief_ray.z
        if be.size(x) != 1:
            raise ValueError("Chief ray cannot be determined. It must be traced alone.")
        R = be.sqrt(x**2 + y**2 + (z - self.pupil_z) ** 2)
        return x, y, z, R.item()


class CentroidReferenceSphereStrategy(ReferenceStrategy):
    """Wavefront analysis strategy using a centroid-anchored reference sphere.

    This strategy computes the wavefront error relative to a reference sphere
    whose center is fixed at the centroid of ray intersections with the image
    surface. The radius is determined by fitting the wavefront points with
    this fixed center.

    This method is robust for:
      * Off-axis fields
      * Asymmetric pupils
      * Systems with obscurations
      * Arbitrary optical configurations (no reliance on paraxial tracing)

    Args:
        optic: The optical system under analysis.
        distribution: The pupil sampling distribution.
        robust_trim_std: Number of standard deviations for optional
            outlier trimming in centroid computation. Set <= 0 to disable.
    """

    def __init__(
        self,
        optic: Optic,
        distribution: BaseDistribution,
        robust_trim_std: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__(optic, distribution, **kwargs)
        self.robust_trim_std = robust_trim_std

    def compute_wavefront_data(
        self, field: tuple[float, float], wavelength: float
    ) -> WavefrontData:
        """Computes wavefront data using a centroid-anchored reference sphere.

        Args:
            field: Tuple (Hx, Hy) of field coordinates.
            wavelength: Wavelength for the analysis in the system's units.

        Returns:
            WavefrontData: Structured data for the computed wavefront.
        """
        # 1. Trace ray bundle to image surface
        rays = self.optic.trace(*field, wavelength, None, self.distribution)

        # 2. Tilt correction in object space (assures rays have identical starting OPL)
        rays.opd = self._correct_tilt(field, rays.opd)

        # 3. Determine reference sphere center and radius
        center_x, center_y, center_z, radius = self._calculate_reference_sphere(rays)

        # 4. Compute OPD from image surface to reference sphere
        opd_img = self._opd_image_to_xp(
            rays, center_x, center_y, center_z, radius, wavelength
        )
        opd = rays.opd - opd_img

        # 5. Remove piston by subtracting mean OPD
        valid_mask = rays.i > 0
        if be.any(valid_mask):
            mean_opd = be.mean(opd[valid_mask])
        else:
            raise ValueError(
                "No valid rays with non-zero intensity for OPD calculation."
            )
        opd_waves = (mean_opd - opd) / (wavelength * 1e-3)  # wavelength: µm to mm

        # 6. Compute pupil coordinates (intersection with reference sphere)
        t = opd_img / self.n_image
        pupil_x = rays.x - t * rays.L
        pupil_y = rays.y - t * rays.M
        pupil_z = rays.z - t * rays.N

        return WavefrontData(
            pupil_x=pupil_x,
            pupil_y=pupil_y,
            pupil_z=pupil_z,
            opd=opd_waves,
            intensity=rays.i,
            radius=radius,
        )

    def _points_from_rays(self, rays: RealRays) -> tuple[be.ndarray, be.ndarray]:
        """Convert ray data to 3D wavefront points.

        Args:
            rays: Traced rays at the image surface.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (points, valid_mask)
        """
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

    def _calculate_reference_sphere(
        self, rays: RealRays
    ) -> tuple[float, float, float, float]:
        """Computes center and radius for the centroid-anchored reference sphere.

        Args:
            rays: Traced rays at the image surface.

        Returns:
            Tuple[float, float, float, float]: (center_x, center_y, center_z, radius)
        """
        wavefront_points, valid_mask = self._points_from_rays(rays)

        # Image-plane intersection points
        image_points = be.stack((rays.x, rays.y, rays.z), axis=1)[valid_mask]

        # Initialize weights
        intensity = rays.i
        weights = intensity[valid_mask]
        weights = be.where(weights < 0.0, 0.0, weights)  # Clamp negatives
        total_weight = be.sum(weights)
        if total_weight == 0:
            weights = be.ones_like(weights)
            total_weight = be.sum(weights)
        else:
            weights = be.ones((image_points.shape[0],))
            total_weight = be.sum(weights)

        # Weighted centroid of image-plane points
        centroid = be.sum(image_points * weights[:, None], axis=0) / total_weight

        # Optional robust trimming to remove extreme outliers in centroid
        if self.robust_trim_std and self.robust_trim_std > 0:
            distances_img = be.linalg.norm(image_points - centroid, axis=1)
            mean_d = be.mean(distances_img)
            std_d = be.std(distances_img)
            if std_d > 0:
                keep_mask = distances_img <= (mean_d + self.robust_trim_std * std_d)
                if be.sum(keep_mask) >= 4:
                    weights = weights * be.array(keep_mask)
                    total_weight = be.sum(weights)
                    centroid = (
                        be.sum(image_points * weights[:, None], axis=0) / total_weight
                    )

        # Radius: weighted mean distance from fixed centroid to wavefront points
        distances_wf = be.linalg.norm(wavefront_points - centroid, axis=1)
        radius = float(be.sum(weights * distances_wf) / be.sum(weights))

        return float(centroid[0]), float(centroid[1]), float(centroid[2]), radius


class BestFitSphereStrategy(CentroidReferenceSphereStrategy):
    """Wavefront analysis strategy using a best-fit reference sphere.

    This strategy computes the wavefront error relative to a reference sphere
    that is determined by a least-squares fit to the wavefront points.
    Unlike its parent class, `CentroidReferenceSphereStrategy`, it does not
    fix the sphere's center at the image-plane centroid, but rather finds
    the optimal center and radius simultaneously.

    Note that this method ignores the location of the image surface. It
    is not sensitive to defocus, tilt or piston effects, as these are naturally
    removed in the fitting process.

    This method is generally more accurate for systems with significant
    wavefront aberrations where the center of curvature deviates from the
    image centroid.
    """

    def __init__(self, optic: Optic, distribution: BaseDistribution, **kwargs) -> None:
        """Initializes the BestFitSphereStrategy.

        Args:
            optic: The optical system to analyze.
            distribution: The pupil sampling distribution.
        """
        super().__init__(optic, distribution, **kwargs)
        self.center = None

    def _calculate_reference_sphere(
        self, rays: RealRays
    ) -> tuple[float, float, float, float]:
        """Computes the best-fit sphere (center and radius) for the wavefront.

        This method solves a linear least-squares problem to find the sphere
        that best fits the 3D wavefront points derived from the ray data.

        Args:
            rays: Traced rays at the image surface.

        Returns:
            Tuple[float, float, float, float]: (center_x, center_y, center_z, radius)
        """
        wavefront_points, _ = self._points_from_rays(rays)

        # Ensure there are enough points for a stable fit
        if wavefront_points.shape[0] < 4:
            raise ValueError("Need at least 4 valid ray samples for a best-fit sphere.")

        x = wavefront_points[:, 0]
        y = wavefront_points[:, 1]
        z = wavefront_points[:, 2]

        # Set up the linear system Ax = b to solve for the sphere parameters.
        # The equation of a sphere is (x-xc)^2 + (y-yc)^2 + (z-zc)^2 = R^2.
        # This can be rearranged into a linear form:
        # 2*x*xc + 2*y*yc + 2*z*zc + (R^2 - xc^2 - yc^2 - zc^2) = x^2 + y^2 + z^2
        A = be.stack([x, y, z, be.ones_like(x)], axis=1)
        b = x**2 + y**2 + z**2

        # Solve the system using a numerically stable least-squares solver.
        try:
            c, _, _, _ = be.linalg.lstsq(A, b, rcond=None)
        except be.linalg.LinAlgError as e:
            raise RuntimeError(f"Least-squares sphere fit failed: {e}") from e

        # Extract the sphere's center (xc, yc, zc) and radius R from the solution.
        xc = c[0] / 2
        yc = c[1] / 2
        zc = c[2] / 2
        radius = be.sqrt(c[3] + xc**2 + yc**2 + zc**2)

        # Save the computed center as an attribute.
        self.center = (float(xc), float(yc), float(zc))

        return self.center[0], self.center[1], self.center[2], float(radius)


STRATEGIES: dict[WavefrontStrategyType, type[ReferenceStrategy]] = {
    "chief_ray": ChiefRayStrategy,
    "centroid_sphere": CentroidReferenceSphereStrategy,
    "best_fit_sphere": BestFitSphereStrategy,
}


def create_strategy(
    strategy_name: WavefrontStrategyType,
    optic: Optic,
    distribution: BaseDistribution,
    **kwargs,
) -> ReferenceStrategy:
    """Factory function to create a wavefront calculation strategy.

    Args:
        strategy_name (str): The name of the strategy ("chief_ray", "centroid_sphere",
            "best_fit_sphere").
        optic (Optic): The optical system.
        distribution (Distribution): The pupil sampling distribution.

    Returns:
        ReferenceStrategy: An instance of the requested strategy.

    Raises:
        ValueError: If the strategy_name is unknown.
    """
    strategy_class = STRATEGIES.get(strategy_name)

    if strategy_class:
        return strategy_class(optic, distribution, **kwargs)
    else:
        raise ValueError(f"Unknown wavefront strategy: {strategy_name}")
