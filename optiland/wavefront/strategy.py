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
from .reference_geometry import PlanarReference, ReferenceGeometry, SphericalReference
from .wavefront_data import WavefrontData

if TYPE_CHECKING:
    from optiland._types import BEArrayT
    from optiland.distribution import BaseDistribution
    from optiland.optic.optic import Optic
    from optiland.rays.real_rays import RealRays

WavefrontStrategyType = Literal["chief_ray", "centroid", "best_fit"]
ReferenceType = Literal["sphere", "plane"]


class ReferenceStrategy(ABC):
    """Abstract base class for a wavefront calculation strategy.

    This class defines the common interface for all wavefront calculation
    strategies. It also provides shared utility methods that concrete
    strategies can use, such as calculating the OPD from the image to the
    exit pupil reference sphere.

    Args:
        optic (Optic): The optical system to analyze.
        distribution (Distribution): The pupil sampling distribution.
        reference_type (str): The type of reference geometry ("sphere" or "plane").
    """

    def __init__(
        self,
        optic: Optic,
        distribution: BaseDistribution,
        reference_type: ReferenceType = "sphere",
        **kwargs,
    ) -> None:
        self.optic = optic
        self.distribution = distribution
        self.reference_type = reference_type
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

    @abstractmethod
    def _create_reference_geometry(self, rays: RealRays) -> ReferenceGeometry:
        """Creates the reference geometry based on the traced rays.

        Args:
            rays: The traced rays at the image surface.

        Returns:
            ReferenceGeometry: The computed reference geometry.
        """
        pass

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
        self._chief_ray = None  # Cache for single field calculation usage

    def compute_wavefront_data(
        self, field: tuple[float, float], wavelength: float
    ) -> WavefrontData:
        """Computes wavefront data using the chief ray reference method.

        Args:
            field (tuple[float, float]): The field coordinates to analyze.
            wavelength (float): The wavelength to use for the analysis.

        Returns:
            WavefrontData: A data object containing the results.
        """
        # 1. Trace chief ray and determine reference sphere
        self._chief_ray = self.optic.trace_generic(
            *field, Px=0.0, Py=0.0, wavelength=wavelength
        )
        geometry = self._create_reference_geometry(self._chief_ray)

        # 2. Calculate reference OPD from the chief ray
        opd_img_ref = geometry.path_length(self._chief_ray, self.n_image)
        opd_ref = self._chief_ray.opd - opd_img_ref
        opd_ref = self._correct_tilt(field, opd_ref, x=0, y=0)

        # 3. Trace the full grid of rays for the field
        rays = self.optic.trace(*field, wavelength, None, self.distribution)
        intensity = self.optic.surface_group.intensity[-1, :]

        # 4. Compute OPD for all rays
        opd_img = geometry.path_length(rays, self.n_image)
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
            radius=geometry.radius,
        )

    def _create_reference_geometry(self, rays: RealRays) -> ReferenceGeometry:
        """Creates reference geometry from cached chief ray."""
        x, y, z = rays.x, rays.y, rays.z
        if be.size(x) != 1:
            raise ValueError("Chief ray cannot be determined. It must be traced alone.")

        if self.reference_type == "sphere":
            return self._create_spherical_ref(x, y, z)
        elif self.reference_type == "plane":
            return self._create_planar_ref(x, y, z, rays.L, rays.M, rays.N)
        else:
            raise ValueError(f"Unknown reference type: {self.reference_type}")

    def _create_spherical_ref(
        self, x: BEArrayT, y: BEArrayT, z: BEArrayT
    ) -> SphericalReference:
        """Create a spherical reference geometry."""
        R = be.sqrt(x**2 + y**2 + (z - self.pupil_z) ** 2)
        return SphericalReference((float(x), float(y), float(z)), R.item())

    def _create_planar_ref(
        self,
        x: BEArrayT,
        y: BEArrayT,
        z: BEArrayT,
        L: BEArrayT,
        M: BEArrayT,
        N: BEArrayT,
    ) -> PlanarReference:
        """Create a planar reference geometry."""
        return PlanarReference(
            (float(x), float(y), float(z)), (float(L), float(M), float(N))
        )


class CentroidStrategy(ReferenceStrategy):
    """Wavefront analysis strategy using a centroid-anchored reference.

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
        """Computes wavefront data using a centroid-anchored reference.

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

        # 3. Determine reference geometry
        geometry = self._create_reference_geometry(rays)

        # 4. Compute OPD from image surface to reference geometry
        opd_img = geometry.path_length(rays, self.n_image)
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

        # 6. Compute pupil coordinates (intersection with reference sphere/plane)
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
            radius=geometry.radius,
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
            raise ValueError("No valid ray samples found for best-fit geometry.")

        p = be.stack((rays.x, rays.y, rays.z), axis=1)[valid]
        d = be.stack((rays.L, rays.M, rays.N), axis=1)[valid]
        s = rays.opd[valid] / self.n_image
        pts = p - s[:, None] * d
        return pts, valid

    def _calculate_weights(
        self, rays: RealRays, image_points: be.ndarray, valid_mask: be.ndarray
    ) -> be.ndarray:
        # Initialize weights
        intensity = rays.i
        weights = intensity[valid_mask]
        weights = be.where(weights < 0.0, 0.0, weights)  # Clamp negatives
        total_weight = be.sum(weights)
        if total_weight == 0:
            weights = be.ones_like(weights)
            total_weight = be.sum(weights)

        # Robust trimming logic
        if self.robust_trim_std and self.robust_trim_std > 0:
            # Centroid for internal trimming calc
            temp_centroid = (
                be.sum(image_points * weights[:, None], axis=0) / total_weight
            )
            distances_img = be.linalg.norm(image_points - temp_centroid, axis=1)
            mean_d = be.mean(distances_img)
            std_d = be.std(distances_img)
            if std_d > 0:
                keep_mask = distances_img <= (mean_d + self.robust_trim_std * std_d)
                if be.sum(keep_mask) >= 4:
                    weights = weights * be.array(keep_mask)

        return weights

    def _create_reference_geometry(self, rays: RealRays) -> ReferenceGeometry:
        wavefront_points, valid_mask = self._points_from_rays(rays)
        image_points = be.stack((rays.x, rays.y, rays.z), axis=1)[valid_mask]

        weights = self._calculate_weights(rays, image_points, valid_mask)
        total_weight = be.sum(weights)

        centroid = be.sum(image_points * weights[:, None], axis=0) / total_weight

        if self.reference_type == "sphere":
            return self._create_spherical_ref(wavefront_points, centroid, weights)
        elif self.reference_type == "plane":
            return self._create_planar_ref(centroid, rays, valid_mask, weights)
        else:
            raise ValueError(f"Unknown reference type: {self.reference_type}")

    def _create_spherical_ref(
        self, wavefront_points: be.ndarray, centroid: be.ndarray, weights: be.ndarray
    ) -> SphericalReference:
        distances_wf = be.linalg.norm(wavefront_points - centroid, axis=1)
        radius = float(be.sum(weights * distances_wf) / be.sum(weights))
        return SphericalReference(
            (float(centroid[0]), float(centroid[1]), float(centroid[2])), radius
        )

    def _create_planar_ref(
        self,
        centroid: be.ndarray,
        rays: RealRays,
        valid_mask: be.ndarray,
        weights: be.ndarray,
    ) -> PlanarReference:
        L, M, N = rays.L[valid_mask], rays.M[valid_mask], rays.N[valid_mask]
        directions = be.stack((L, M, N), axis=1)
        mean_direction = be.sum(directions * weights[:, None], axis=0) / be.sum(weights)
        norm = be.linalg.norm(mean_direction)
        if norm > 0:
            mean_direction = mean_direction / norm

        return PlanarReference(
            (float(centroid[0]), float(centroid[1]), float(centroid[2])),
            (
                float(mean_direction[0]),
                float(mean_direction[1]),
                float(mean_direction[2]),
            ),
        )


class BestFitStrategy(CentroidStrategy):
    """Wavefront analysis strategy using a best-fit reference geometry.

    This strategy computes the wavefront error relative to a reference
    that is determined by a least-squares fit to the wavefront points.
    """

    def __init__(self, optic: Optic, distribution: BaseDistribution, **kwargs) -> None:
        super().__init__(optic, distribution, **kwargs)
        self.center = None

    def _create_reference_geometry(self, rays: RealRays) -> ReferenceGeometry:
        wavefront_points, _ = self._points_from_rays(rays)

        if wavefront_points.shape[0] < 4:
            raise ValueError("Need at least 4 valid ray samples for best-fit.")

        x = wavefront_points[:, 0]
        y = wavefront_points[:, 1]
        z = wavefront_points[:, 2]

        if self.reference_type == "sphere":
            return self._create_spherical_ref(x, y, z)
        elif self.reference_type == "plane":
            return self._create_planar_ref(wavefront_points)
        else:
            raise ValueError(f"Unknown reference type: {self.reference_type}")

    def _create_spherical_ref(
        self, x: be.ndarray, y: be.ndarray, z: be.ndarray
    ) -> SphericalReference:
        # Sphere fit
        A = be.stack([x, y, z, be.ones_like(x)], axis=1)
        b = x**2 + y**2 + z**2
        try:
            c, _, _, _ = be.linalg.lstsq(A, b, rcond=None)
        except be.linalg.LinAlgError as e:
            raise RuntimeError(f"Least-squares sphere fit failed: {e}") from e

        xc = c[0] / 2
        yc = c[1] / 2
        zc = c[2] / 2
        radius = be.sqrt(c[3] + xc**2 + yc**2 + zc**2)
        self.center = (float(xc), float(yc), float(zc))
        return SphericalReference(self.center, float(radius))

    def _create_planar_ref(self, wavefront_points: be.ndarray) -> PlanarReference:
        # Plane fit: Ax + By + Cz + D = 0
        # Center data
        centroid = be.mean(wavefront_points, axis=0)
        centered_points = wavefront_points - centroid

        # SVD
        u, s, vh = be.linalg.svd(centered_points, full_matrices=False)
        # Normal is the last row of vh (corresponding to smallest singular value)
        normal = vh[-1, :]

        return PlanarReference(
            (float(centroid[0]), float(centroid[1]), float(centroid[2])),
            (float(normal[0]), float(normal[1]), float(normal[2])),
        )


STRATEGIES: dict[WavefrontStrategyType, type[ReferenceStrategy]] = {
    "chief_ray": ChiefRayStrategy,
    "centroid_sphere": CentroidStrategy,  # Kept for backward compat
    "centroid": CentroidStrategy,
    "best_fit_sphere": BestFitStrategy,  # Kept for backward compat
    "best_fit": BestFitStrategy,
}


def create_strategy(
    strategy_name: WavefrontStrategyType,
    optic: Optic,
    distribution: BaseDistribution,
    reference_type: ReferenceType = "sphere",
    **kwargs,
) -> ReferenceStrategy:
    """Factory function to create a wavefront calculation strategy.

    Args:
        strategy_name (str): The name of the strategy ("chief_ray", "centroid",
            "best_fit").
        optic (Optic): The optical system.
        distribution (Distribution): The pupil sampling distribution.
        reference_type (str): "sphere" or "plane".

    Returns:
        ReferenceStrategy: An instance of the requested strategy.

    Raises:
        ValueError: If the strategy_name is unknown.
    """
    strategy_class = STRATEGIES.get(strategy_name)

    if strategy_class:
        return strategy_class(
            optic, distribution, reference_type=reference_type, **kwargs
        )
    else:
        raise ValueError(f"Unknown wavefront strategy: {strategy_name}")
