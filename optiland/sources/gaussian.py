"""
Gaussian Source Module

This module implements a Gaussian source that generates rays based on a Gaussian
intensity profile in both spatial and angular domains. The source uses quasi-random
Sobol sequences for improved sampling quality and differentiable ray generation.
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

    The source supports both symmetric and astigmatic beam profiles, making it
    suitable for modeling various sources from single-mode fibers to diode lasers.

    Attributes:
        wavelength (float): Wavelength in micrometers (µm)
        total_power (float): Total optical power in Watts (W)
        s_x_um (float): Spatial 1/e² radius in X direction (µm)
        s_y_um (float): Spatial 1/e² radius in Y direction (µm)
        s_L_rad (float): Angular 1/e² radius in L direction (radians)
        s_M_rad (float): Angular 1/e² radius in M direction (radians)
        sigma_spatial_mm_x (float): Spatial sampling sigma in X (mm)
        sigma_spatial_mm_y (float): Spatial sampling sigma in Y (mm)
        sigma_angular_rad_x (float): Angular sampling sigma in L (radians)
        sigma_angular_rad_y (float): Angular sampling sigma in M (radians)
    """

    def __init__(
        self,
        wavelength: float = 1.55,
        total_power: float = 1.0,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mfd: float = None,
        spatial_width_x: float = None,
        spatial_width_y: float = None,
        angular_width_x: float = None,
        angular_width_y: float = None,
    ):
        """
        Initialize the GaussianSource with flexible beam parameters.

        This constructor allows for modeling both symmetric and astigmatic beams.
        You can specify beam parameters in multiple ways:
        1. Use 'mfd' for a simple circular, diffraction-limited beam
        2. Use 'spatial_width_x/y' for custom spatial sizes (diffraction-limited)
        3. Use both spatial and angular widths for non-diffraction-limited beams

        Args:
            wavelength (float): Wavelength of the source in micrometers (µm).
                Default is 1.55.
            total_power (float): Total optical power of the source in Watts (W).
                Default is 1.0.
            position (tuple[float, float, float]): The (x, y, z) position of
                the source in millimeters. Defaults to (0.0, 0.0, 0.0).
            mfd (float, optional): Mode Field Diameter in micrometers (µm).
                Creates a circular, diffraction-limited beam. Cannot be used
                with spatial_width parameters.
            spatial_width_x (float, optional): Spatial 1/e² radius (beam waist, w₀)
                in X direction, in micrometers (µm).
            spatial_width_y (float, optional): Spatial 1/e² radius (beam waist, w₀)
                in Y direction, in micrometers (µm). If not provided, defaults
                to spatial_width_x for circular beams.
            angular_width_x (float, optional): Angular 1/e² radius (far-field
                divergence, θ₀) in X direction, in radians. If not provided,
                calculated assuming diffraction-limited propagation.
            angular_width_y (float, optional): Angular 1/e² radius (far-field
                divergence, θ₀) in Y direction, in radians. If not provided,
                calculated assuming diffraction-limited propagation.

        Raises:
            ValueError: If insufficient parameters are provided to determine
                beam size, or if conflicting parameter combinations are used.
        """
        super().__init__(position=position)

        self.wavelength = wavelength
        self.total_power = total_power

        # --- Parameter Validation and Processing ---

        # Step A: Initial validation
        has_mfd = mfd is not None
        has_spatial = (spatial_width_x is not None) or (spatial_width_y is not None)

        if not (has_mfd or has_spatial):
            raise ValueError(
                "Must provide at least one sizing parameter: 'mfd' or "
                "'spatial_width_x/spatial_width_y'"
            )

        if has_mfd and has_spatial:
            raise ValueError(
                "Cannot specify both 'mfd' and 'spatial_width' parameters. "
                "Use 'mfd' for simple circular beams or 'spatial_width_x/y' "
                "for custom beam sizes."
            )

        # Step B: Determine spatial widths (s_x, s_y)
        if spatial_width_x is not None:
            s_x_um = spatial_width_x
        elif mfd is not None:
            s_x_um = mfd / 2.0
        else:
            raise ValueError("Cannot determine spatial width in X")

        if spatial_width_y is not None:
            s_y_um = spatial_width_y
        elif s_x_um is not None:
            s_y_um = s_x_um  # Default to circular beam
        else:
            raise ValueError("Cannot determine spatial width in Y")

        # Step C: Determine angular widths (s_L, s_M)
        if angular_width_x is not None:
            s_L_rad = angular_width_x
        else:
            # Calculate from spatial width (diffraction-limited case)
            s_L_rad = self.wavelength / (be.pi * s_x_um)

        if angular_width_y is not None:
            s_M_rad = angular_width_y
        else:
            # Calculate from spatial width (diffraction-limited case)
            s_M_rad = self.wavelength / (be.pi * s_y_um)

        # Step D: Store beam parameters and calculate sampling sigmas
        self.s_x_um = s_x_um
        self.s_y_um = s_y_um
        self.s_L_rad = s_L_rad
        self.s_M_rad = s_M_rad

        # Convert units and calculate sigmas for importance sampling
        # For a luminance profile ~exp(-2*x²/s_x²), the sampling sigma is s_x/2
        self.sigma_spatial_mm_x = (s_x_um * 1e-3) / 2.0
        self.sigma_spatial_mm_y = (s_y_um * 1e-3) / 2.0
        self.sigma_angular_rad_x = s_L_rad / 2.0
        self.sigma_angular_rad_y = s_M_rad / 2.0

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

        # Spatial coordinates (x, y) - using separate sigmas for X and Y
        x_start = self.sigma_spatial_mm_x * sqrt2 * be.erfinv(2 * u_samples[:, 0] - 1)
        y_start = self.sigma_spatial_mm_y * sqrt2 * be.erfinv(2 * u_samples[:, 1] - 1)
        z_start = be.zeros_like(x_start)

        # Angular coordinates (L, M) - now using separate sigmas for X and Y
        L_initial = (
            self.sigma_angular_rad_x * sqrt2 * be.erfinv(2 * u_samples[:, 2] - 1)
        )
        M_initial = (
            self.sigma_angular_rad_y * sqrt2 * be.erfinv(2 * u_samples[:, 3] - 1)
        )

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

        # --- Calculate power and intensity arrays ---
        power_per_ray = self.total_power / num_valid_rays

        # Calculate theoretical Gaussian intensity profile for visualization
        # This shows the luminance profile even though each ray carries equal power

        # Convert spatial coordinates back to micrometers for luminance calculation
        x_start_um = x_start * 1000  # mm to μm
        y_start_um = y_start * 1000  # mm to μm

        # astigmatic Gaussian luminance profile
        # L(x,y,L,M) = L0 * exp(-2*x²/s_x² - 2*y²/s_y² - 2*L²/s_L² - 2*M²/s_M²)
        term_x = -2.0 * (x_start_um / self.s_x_um) ** 2
        term_y = -2.0 * (y_start_um / self.s_y_um) ** 2
        term_L = -2.0 * (L_initial / self.s_L_rad) ** 2
        term_M = -2.0 * (M_initial / self.s_M_rad) ** 2

        # Theoretical Gaussian intensity profile (for visualization)
        theoretical_intensity = be.exp(term_x + term_y + term_L + term_M)

        # Scale the theoretical intensity to match the total power
        # This gives us both: equal power per ray AND correct intensity profile
        # for visualization
        intensity_power_array = theoretical_intensity * (
            power_per_ray / be.mean(theoretical_intensity)
        )

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

        # Transform rays from local source coordinates to global coordinates
        self.cs.globalize(rays)

        return rays

    def __repr__(self) -> str:
        """Return a string representation of the GaussianSource."""
        position = (float(self.cs.x), float(self.cs.y), float(self.cs.z))

        # Show the beam parameters that define this source
        beam_params = []

        # If it's a symmetric beam, show as effective MFD
        if (
            abs(self.s_x_um - self.s_y_um) < 1e-6  # Spatial symmetry
            and abs(self.s_L_rad - self.s_M_rad) < 1e-9
        ):  # Angular symmetry
            eff_mfd = 2.0 * self.s_x_um
            beam_params.append(f"mfd_eff={eff_mfd:.2f}µm")
        else:
            # Asymmetric beam - show individual parameters
            beam_params.append(f"spatial=({self.s_x_um:.2f}×{self.s_y_um:.2f})µm")
            beam_params.append(f"angular=({self.s_L_rad:.4f}×{self.s_M_rad:.4f})rad")

        beam_str = ", ".join(beam_params)

        return (
            f"GaussianSource({beam_str}, wavelength={self.wavelength}µm, "
            f"total_power={self.total_power}W, position={position})"
        )
