"""
This module defines the reference sphere calculation strategies.

Kramer Harrison, 2024
"""

from abc import ABC, abstractmethod

import optiland.backend as be


class ReferenceSphereCalculator(ABC):
    """
    Abstract base class for reference sphere calculation strategies.
    """

    def __init__(self, optic):
        self.optic = optic

    @abstractmethod
    def calculate(self, pupil_z):
        """
        Calculate the reference sphere parameters.

        Args:
            pupil_z (float): The z-coordinate of the exit pupil.

        Returns:
            tuple: A tuple containing the reference sphere parameters
                   (xc, yc, zc, R).
        """
        pass


class ChiefRayReferenceSphereCalculator(ReferenceSphereCalculator):
    """
    Calculates the reference sphere based on the chief ray.
    """

    def __init__(self, optic):
        super().__init__(optic)

        # Z location of the exit pupil
        self.pupil_z = optic.paraxial.XPL() + optic.surface_group.positions[-1]

    def calculate(self):
        """
        Determine reference sphere center and radius from chief ray.
        """
        x = self.optic.surface_group.x[-1, :]
        y = self.optic.surface_group.y[-1, :]
        z = self.optic.surface_group.z[-1, :]
        if be.size(x) != 1:
            raise ValueError("Chief ray cannot be determined. It must be traced alone.")
        R = be.sqrt(x**2 + y**2 + (z - self.pupil_z) ** 2)
        return x, y, z, R


class BestFitReferenceSphereCalculator(ReferenceSphereCalculator):
    """Calculates the best-fit reference sphere using OPD-based wavefront fitting."""

    def __init__(self, optic):
        """Initialize the calculator.

        Args:
            optic: Optical system instance containing traced ray data.
        """
        super().__init__(optic)
        self.n_image = optic.n()[-1]  # Refractive index at the image surface

    def calculate(self):
        """Fit a best-fit reference sphere from ray data.

        Returns:
            tuple: (xc, yc, zc, R) where (xc, yc, zc) is the sphere center
                in image-space coordinates and R is the sphere radius.
        """
        pts, _ = self._points_from_rays(self.optic.rays)
        center, radius = self._fit_sphere(pts)
        return *center, radius

    def _points_from_rays(self, rays):
        """Convert ray data into 3D wavefront points in image space.

        Args:
            rays: Object with arrays x, y, z, L, M, N, opd.

        Returns:
            tuple:
                pts (ndarray): (N,3) wavefront points.
                mask (ndarray): Boolean mask of valid rays.
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
            raise ValueError("No valid ray samples found")

        p = be.stack((rays.x, rays.y, rays.z), axis=1)[valid]
        d = be.stack((rays.L, rays.M, rays.N), axis=1)[valid]

        # displacement along rays to reach constant-OPL wavefront surface
        s = rays.opd[valid] / self.n_image
        pts = p - s[:, None] * d
        return pts, valid

    def _fit_sphere(self, pts):
        """Fit a sphere to points using a stable linear algebraic method.

        Args:
            pts (ndarray): (N,3) array of 3D points.

        Returns:
            tuple:
                center (ndarray): Sphere center (3,).
                radius (float): Sphere radius.
        """
        pts = be.array(pts)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("pts must be (N,3)")

        # Normalize for numerical stability
        mean = pts.mean(axis=0)
        scale = be.ptp(pts, axis=0).max()
        scale = scale if scale > 0 else 1.0
        q = (pts - mean) / scale

        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        A = be.column_stack((x, y, z, be.ones_like(x)))
        b = -(x**2 + y**2 + z**2)

        coef, *_ = be.linalg.lstsq(A, b, rcond=None)
        center_q = -0.5 * coef[:3]
        radius_q = be.sqrt(be.dot(center_q, center_q) - coef[3])

        center = scale * center_q + mean
        radius = scale * radius_q
        return center, float(radius)


def create_reference_sphere_calculator(strategy, optic):
    """
    Factory function to create a reference sphere calculator.

    Args:
        strategy (str): The name of the strategy to use. "chief_ray" or "best_fit".
        optic (Optic): The optical system.

    Returns:
        ReferenceSphereCalculator: An instance of the requested calculator.
    """
    if strategy == "chief_ray":
        return ChiefRayReferenceSphereCalculator(optic)
    elif strategy == "best_fit":
        return BestFitReferenceSphereCalculator(optic)
    else:
        raise ValueError(f"Unknown reference sphere strategy: {strategy}")
