"""
Collimated Gaussian Source Module

This module implements a collimated Gaussian source that generates rays with a
Gaussian spatial distribution but collimated direction (all rays parallel to the
optical axis). This source uses quasi-random Sobol sequences for improved sampling
quality and differentiable ray generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland import backend as be
from optiland.sources.base import BaseSource

if TYPE_CHECKING:
    from optiland.rays import RealRays


class CollimatedGaussianSource(BaseSource):
    """
    A collimated Gaussian source that generates rays with Gaussian spatial distribution
    and collimated propagation direction.

    This class generates rays based on a Gaussian spatial distribution while ensuring
    all rays propagate parallel to the optical axis (collimated). It uses quasi-random
    Sobol sequences for improved sampling quality and backend-agnostic operations for
    differentiability.

    The source models a collimated Gaussian beam with specified beam waist, wavelength,
    and total power.

    Attributes:
        gaussian_waist (float): Gaussian beam waist in millimeters (mm)
        wavelength (float): Wavelength in micrometers (µm)
        total_power (float): Total optical power in Watts (W)
        sigma_spatial_mm (float): Spatial sigma for sampling in mm
    """

    def __init__(
        self,
        gaussian_waist: float = 0.0052,  # mm, half of 10.4 µm MFD
        wavelength: float = 1.55,  # µm
        total_power: float = 1.0,  # W
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """
        Initialize the CollimatedGaussianSource with source-specific parameters.

        Args:
            gaussian_waist (float): Gaussian beam waist in millimeters (mm).
                Default is 5.2 mm.
            wavelength (float): Wavelength of the source in micrometers (µm).
                Default is 1.55.
            total_power (float): Total optical power of the source in Watts (W).
                Default is 1.0.
            position (tuple[float, float, float]): The (x, y, z) position of
                the source in millimeters. Defaults to (0.0, 0.0, 0.0).
        """
        super().__init__(position=position)

        self.gaussian_waist = gaussian_waist
        self.wavelength = wavelength
        self.total_power = total_power

        # --- Sigma for Gaussian spatial sampling ---
        # For a Gaussian profile ~exp(-2*x^2/w^2), the sampling sigma is w/2
        self.sigma_spatial_mm = self.gaussian_waist / 2.0

    def generate_rays(self, num_rays: int) -> RealRays:
        """
        Generate collimated rays using backend-agnostic quasi-random Sobol sampling.

        This method uses Sobol sequences for improved sampling quality and applies
        the inverse error function to convert uniform samples to Gaussian spatial
        distributions. All rays are collimated (parallel to optical axis).

        Args:
            num_rays (int): The number of rays to generate

        Returns:
            RealRays: An object containing the generated rays
        """
        from optiland.rays import RealRays

        # Generate quasi-random samples using Sobol sequences
        # We need 2 dimensions for spatial coordinates: x, y
        u_samples = be.sobol_sampler(dim=2, num_samples=num_rays, scramble=True)

        # Convert uniform samples to Gaussian using inverse error function
        sqrt2 = be.sqrt(be.array(2.0))

        # Spatial coordinates (x, y) with Gaussian distribution
        x_start = self.sigma_spatial_mm * sqrt2 * be.erfinv(2 * u_samples[:, 0] - 1)
        y_start = self.sigma_spatial_mm * sqrt2 * be.erfinv(2 * u_samples[:, 1] - 1)
        z_start = be.zeros_like(x_start)

        # Collimated direction: all rays parallel to optical axis (z-direction)
        L_initial = be.zeros_like(x_start)
        M_initial = be.zeros_like(x_start)
        N_initial = be.ones_like(x_start)

        num_valid_rays = be.size(L_initial)

        print(
            f"Generated {num_valid_rays} collimated rays for simulation using "
            f"backend: '{be.get_backend()}'"
        )

        # --- Calculate power per ray (corrected for importance sampling) ---
        # Since rays are already importance-sampled according to Gaussian distribution,
        # each ray should carry equal power to maintain energy conservation
        power_per_ray = self.total_power / num_valid_rays
        intensity_power_array = be.full((num_valid_rays,), power_per_ray)

        # --- Create wavelength array ---
        wavelength_array = be.full((num_valid_rays,), self.wavelength)

        # --- Create the final RealRays object ---
        rays = RealRays(
            x=x_start,
            y=y_start,
            z=z_start,
            L=L_initial,
            M=M_initial,
            N=N_initial,
            intensity=intensity_power_array,
            wavelength=wavelength_array,
        )

        print(
            f"Successfully created backend-agnostic RealRays object with "
            f"{be.size(rays.x)} collimated rays."
        )
        print(
            f"Each ray carries equal power: {power_per_ray:.3e} Watts "
            f"(importance-sampled)."
        )

        # Transform rays from local source coordinates to global coordinates
        self.cs.globalize(rays)

        return rays

    def __repr__(self) -> str:
        """Return a string representation of the CollimatedGaussianSource."""
        position = (float(self.cs.x), float(self.cs.y), float(self.cs.z))
        return (
            f"CollimatedGaussianSource(gaussian_waist={self.gaussian_waist}, "
            f"wavelength={self.wavelength}, total_power={self.total_power}, "
            f"position={position})"
        )
