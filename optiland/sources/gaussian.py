"""
Gaussian Source Module

This module implements a Gaussian source that generates rays based on a Gaussian
intensity profile in both spatial and angular domains. The source uses quasi-random
Sobol sequences for improved sampling quality and differentiable ray generation.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland import backend as be
from optiland.sources.base import BaseSource

if TYPE_CHECKING:
    from optiland.rays import RealRays


class GaussianSource(BaseSource):
    """
    A Gaussian source that generates rays with Gaussian spatial and angular
    distributions.

    This class generates rays based on a Gaussian distribution defined by the source
    parameters. It uses quasi-random Sobol sequences for improved sampling quality
    and backend-agnostic operations for differentiability.

    The source models a fiber-like Gaussian beam with specified mode field diameter,
    wavelength, and total power.

    Attributes:
        mfd (float): Mode Field Diameter in micrometers (µm)
        wavelength (float): Wavelength in micrometers (µm)
        total_power (float): Total optical power in Watts (W)
        sigma_spatial_mm (float): Spatial sigma for sampling in mm
        sigma_angular_rad (float): Angular sigma for sampling in radians
    """

    def __init__(
        self, mfd: float = 10.4, wavelength: float = 1.55, total_power: float = 1.0
    ):
        """
        Initialize the GaussianSource with source-specific parameters.

        Args:
            mfd (float): Mode Field Diameter in micrometers (µm). Default is 10.4.
            wavelength (float): Wavelength of the source in micrometers (µm).
                Default is 1.55.
            total_power (float): Total optical power of the source in Watts (W).
                Default is 1.0.
        """
        self.mfd = mfd
        self.wavelength = wavelength
        self.total_power = total_power

        # --- Derived parameters for Gaussian distribution ---
        # Use be.pi for backend-agnostic pi constant
        w0_um = self.mfd / 2.0
        s_x_um = w0_um  # 1/e^2 spatial radius for luminance in µm
        # 1/e^2 angular radius in L-space (radians)
        s_L_rad = self.wavelength / (be.pi * w0_um)

        # Convert units for Optiland (assuming mm)
        s_x_mm = s_x_um * 1e-3

        # --- Sigmas for Importance Sampling ---
        # For a luminance profile ~exp(-2*x^2/s_x^2), the sampling sigma is s_x/2
        self.sigma_spatial_mm = s_x_mm / 2.0
        self.sigma_angular_rad = s_L_rad / 2.0

    def generate_rays(self, num_rays: int) -> RealRays:
        """
        Generate rays using backend-agnostic quasi-random Sobol sampling.

        This method uses Sobol sequences for improved sampling quality and applies
        the inverse error function to convert uniform samples to Gaussian distributions.

        Args:
            num_rays (int): The number of rays to attempt to generate

        Returns:
            RealRays: An object containing the generated rays

        Raises:
            ValueError: If no valid rays are generated after filtering
        """
        from optiland.rays import RealRays

        # Generate quasi-random samples using Sobol sequences
        # We need 4 dimensions: x, y, L, M
        u_samples = be.sobol_sampler(dim=4, num_samples=num_rays, scramble=True)

        # Convert uniform samples to Gaussian using inverse error function
        sqrt2 = be.sqrt(be.array(2.0))

        # Spatial coordinates (x, y)
        x_start = self.sigma_spatial_mm * sqrt2 * be.erfinv(2 * u_samples[:, 0] - 1)
        y_start = self.sigma_spatial_mm * sqrt2 * be.erfinv(2 * u_samples[:, 1] - 1)
        z_start = be.zeros_like(x_start)

        # Angular coordinates (L, M)
        L_initial = self.sigma_angular_rad * sqrt2 * be.erfinv(2 * u_samples[:, 2] - 1)
        M_initial = self.sigma_angular_rad * sqrt2 * be.erfinv(2 * u_samples[:, 3] - 1)

        # --- Filter for physically possible rays ---
        # Rays must satisfy L² + M² < 1 for physical propagation
        valid_mask = L_initial**2 + M_initial**2 < 1.0

        # Apply the mask to filter valid rays
        x_start = x_start[valid_mask]
        y_start = y_start[valid_mask]
        z_start = z_start[valid_mask]
        L_initial = L_initial[valid_mask]
        M_initial = M_initial[valid_mask]

        num_valid_rays = be.size(L_initial)
        if num_valid_rays == 0:
            raise ValueError(
                "No valid rays generated after filtering. "
                "Check parameters or increase num_rays."
            )

        print(
            f"Generated {num_valid_rays} valid rays for simulation using "
            f"backend: '{be.get_backend()}'"
        )

        # --- Calculate power per ray ---
        power_per_ray = self.total_power / num_valid_rays
        intensity_power_array = be.full((num_valid_rays,), power_per_ray)

        # --- Calculate N component (ensures ray normalization) ---
        N_initial = be.sqrt(
            be.maximum(be.array(0.0), 1.0 - L_initial**2 - M_initial**2)
        )

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
            f"{be.size(rays.x)} rays."
        )
        print(f"Each ray carries {power_per_ray:.3e} Watts of power.")

        return rays

    def __repr__(self) -> str:
        """Return a string representation of the GaussianSource."""
        return (
            f"GaussianSource(mfd={self.mfd}, wavelength={self.wavelength}, "
            f"total_power={self.total_power})"
        )
