"""Gradient-Index (GRIN) propagation model.

This module implements ray propagation through gradient-index media using
the Runge-Kutta 4th order (RK4) numerical integration method.

Jie Cai (zhazhajust), 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.propagation.base import BasePropagationModel

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.rays.real_rays import RealRays


class GRINPropagation(BasePropagationModel):
    """Gradient-Index (GRIN) ray propagation model.

    This model implements ray propagation through gradient-index media
    using numerical integration (RK4 method) to solve the ray equation.

    """

    def __init__(self, material: BaseMaterial):
        """Initializes the GRINPropagation model.

        Args:
            material: A reference to the parent material instance, used to
                query for refractive index and gradient.

        """
        self.material = material
        self.step_size = 0.01  # Default step size for RK4 integration (in mm)

    def propagate(self, rays: RealRays, t: float) -> None:
        """Propagates rays through the GRIN medium to the exit surface.

        This method uses RK4 numerical integration to solve the ray equation
        in a gradient-index medium. The propagation continues until rays
        reach either the exit surface or return to the entrance surface.

        Args:
            rays: The rays object to be propagated.
            t: The approximate distance to the exit surface along the z-axis.
                This defines the initial propagation direction. Rays can exit
                from either the exit surface (z_initial + t) or the entrance
                surface (z_initial) if they undergo internal reflection.

        """
        # For GRIN media, we need to use numerical integration
        # For now, use the provided t as a target distance and integrate
        # through the GRIN medium.

        # Store initial state for each ray
        z_initial = be.copy(rays.z)

        # Target z position (exit surface)
        z_target = z_initial + t

        # Maximum iterations to prevent infinite loops
        max_iterations = 10000

        # Pre-calculate bounds for all rays
        z_min_all = be.minimum(z_initial, z_target)
        z_max_all = be.maximum(z_initial, z_target)

        for _iteration in range(max_iterations):
            # Check which rays still need to propagate
            # Rays are active if they are within the medium bounds
            # Allow rays to propagate in both directions (N can be positive or negative)
            active = (rays.z > z_min_all - 1e-6) & (rays.z < z_max_all + 1e-6)

            if not be.any(active):
                break

            # RK4 integration step for active rays
            self._rk4_step(rays, active, z_min_all[active], z_max_all[active])

        # Normalize direction vectors
        rays.normalize()

    def _rk4_step(self, rays: RealRays, active, z_min: float, z_max: float) -> None:
        """Performs one RK4 integration step for the active rays.

        Args:
            rays: The rays object (modified in-place).
            active: Boolean array indicating which rays are active.
            z_min: Minimum z boundary (entrance surface) for active rays.
            z_max: Maximum z boundary (exit surface) for active rays.

        """
        # Get active ray positions and directions
        x = rays.x[active]
        y = rays.y[active]
        z = rays.z[active]
        L = rays.L[active]
        M = rays.M[active]
        N = rays.N[active]
        w = rays.w[active]

        # Adaptive step size based on distance to boundaries
        # Initialize with default step size
        step_size = be.full_like(z, self.step_size)

        # For rays propagating forward (N > 0)
        forward_mask = N > 0
        if be.any(forward_mask):
            remaining_forward = z_max - z
            # Only apply limit where N > 0 to avoid zero step size for backward rays
            step_size = be.where(
                forward_mask,
                be.minimum(step_size, be.abs(remaining_forward * 0.9)),
                step_size,
            )

        # For rays propagating backward (N < 0)
        backward_mask = N < 0
        if be.any(backward_mask):
            remaining_backward = z - z_min
            step_size = be.where(
                backward_mask,
                be.minimum(step_size, be.abs(remaining_backward * 0.9)),
                step_size,
            )

        # Initial state: [x, y, z, L, M, N]
        # We use the Hamiltonian formulation for GRIN propagation
        # State vector: [x, y, z, px, py, pz] where p = n * direction

        # K1
        k1, n1 = self._ray_derivative(x, y, z, L, M, N, w)

        # K2
        x2 = x + 0.5 * step_size * k1[0]
        y2 = y + 0.5 * step_size * k1[1]
        z2 = z + 0.5 * step_size * k1[2]
        L2 = L + 0.5 * step_size * k1[3]
        M2 = M + 0.5 * step_size * k1[4]
        N2 = N + 0.5 * step_size * k1[5]
        k2, n2 = self._ray_derivative(x2, y2, z2, L2, M2, N2, w)

        # K3
        x3 = x + 0.5 * step_size * k2[0]
        y3 = y + 0.5 * step_size * k2[1]
        z3 = z + 0.5 * step_size * k2[2]
        L3 = L + 0.5 * step_size * k2[3]
        M3 = M + 0.5 * step_size * k2[4]
        N3 = N + 0.5 * step_size * k2[5]
        k3, n3 = self._ray_derivative(x3, y3, z3, L3, M3, N3, w)

        # K4
        x4 = x + step_size * k3[0]
        y4 = y + step_size * k3[1]
        z4 = z + step_size * k3[2]
        L4 = L + step_size * k3[3]
        M4 = M + step_size * k3[4]
        N4 = N + step_size * k3[5]
        k4, n4 = self._ray_derivative(x4, y4, z4, L4, M4, N4, w)

        # Update state
        rays.x[active] = x + (step_size / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        rays.y[active] = y + (step_size / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        rays.z[active] = z + (step_size / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        rays.L[active] = L + (step_size / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        rays.M[active] = M + (step_size / 6.0) * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
        rays.N[active] = N + (step_size / 6.0) * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5])

        # Integate OPD: d(OPD)/ds = n
        # OPD_new = OPD_old + integral(n ds)
        # Using RK4 approximation for the integral
        rays.opd[active] += (step_size / 6.0) * (n1 + 2 * n2 + 2 * n3 + n4)

    def _ray_derivative(self, x, y, z, L, M, N, w):
        """Calculates the derivative of the ray state.

        The ray equation in GRIN media is:
            d/ds(n * dr/ds) = grad(n)

        Expanding: d²r/ds² = (1/n) * (grad(n) - (dr/ds) * (dr/ds · grad(n)))

        Args:
            x, y, z: Ray positions.
            L, M, N: Ray direction cosines.
            w: Wavelength.

        Returns:
            Tuple of derivatives: (dx/ds, dy/ds, dz/ds, dL/ds, dM/ds, dN/ds)
            and refractive index n (for OPD calculation).

        """
        # Get refractive index and gradient at current position
        if hasattr(self.material, "get_index_and_gradient"):
            n, dn_dx, dn_dy, dn_dz = self.material.get_index_and_gradient(x, y, z, w)
        else:
            # Fallback for materials without gradient support
            n = self.material.n(w, x=x, y=y, z=z)
            dn_dx = be.zeros_like(n)
            dn_dy = be.zeros_like(n)
            dn_dz = be.zeros_like(n)

        # Direction vector is (L, M, N)
        # d(position)/ds = direction
        dx_ds = L
        dy_ds = M
        dz_ds = N

        # d(direction)/ds = (1/n) * (grad(n) - direction * (direction · grad(n)))
        dot_product = L * dn_dx + M * dn_dy + N * dn_dz
        dL_ds = (1.0 / n) * (dn_dx - L * dot_product)
        dM_ds = (1.0 / n) * (dn_dy - M * dot_product)
        dN_ds = (1.0 / n) * (dn_dz - N * dot_product)

        return (dx_ds, dy_ds, dz_ds, dL_ds, dM_ds, dN_ds), n

    @classmethod
    def from_dict(cls, d: dict, material: BaseMaterial = None) -> GRINPropagation:
        """Creates a GRINPropagation model from a dictionary.

        Args:
            d: The dictionary representation of the model.
            material: The parent material instance.

        Returns:
            An instance of the GRINPropagation model.

        """
        return cls(material=material)
