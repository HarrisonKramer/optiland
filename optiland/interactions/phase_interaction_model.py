"""
Provides the phase interaction model.
"""

from __future__ import annotations

import typing

from optiland import backend as be
from optiland.interactions.base import BaseInteractionModel
from optiland.phase.base import BasePhaseProfile

if typing.TYPE_CHECKING:
    from optiland.raytrace.rays import ParaxialRays, RealRays
    from optiland.surfaces.standard_surface import Surface


class PhaseInteractionModel(BaseInteractionModel):
    """An interaction model for surfaces with an arbitrary phase profile.

    This model implements the Generalized Snell's Law, where the change in a
    ray's transverse wavevector is determined by the gradient of a phase
    profile defined on the surface. It is a "thin sheet" model and must be
    used with a `Plane` geometry.

    Args:
        parent_surface (Surface): The surface to which this interaction model is
            attached.
        phase_profile (BasePhaseProfile): The phase profile strategy object that
            defines the phase and its gradient.
    """

    interaction_type = "phase"

    def __init__(
        self,
        parent_surface: Surface | None,
        phase_profile: BasePhaseProfile,
        is_reflective: bool,
        **kwargs,
    ):
        super().__init__(parent_surface, is_reflective=is_reflective, **kwargs)
        self.phase_profile = phase_profile

    def interact_real_rays(self, rays: RealRays) -> RealRays:
        from optiland.geometries.plane import Plane

        if self.parent_surface is None:
            raise RuntimeError("Parent surface not set for PhaseInteractionModel.")
        if not isinstance(self.parent_surface.geometry, Plane):
            raise TypeError(
                "PhaseInteractionModel can only be used with Plane geometries."
            )
        # Get incident state
        x, y = rays.x, rays.y
        l_i, m_i, n_i = rays.L, rays.M, rays.N
        rays.L0, rays.M0, rays.N0 = l_i, m_i, n_i
        n1 = self.parent_surface.material_pre.n(rays.w)
        n2 = self.parent_surface.material_post.n(rays.w)
        if self.is_reflective:
            n2 = n1
        k0 = 2 * be.pi / rays.w

        # 1. Get local surface normal (N)
        nx, ny, nz = self.parent_surface.geometry.surface_normal(rays)

        # 2. Get incident wavevector (k_in)
        k_ix = n1 * k0 * l_i
        k_iy = n1 * k0 * m_i
        k_iz = n1 * k0 * n_i

        # 3. Get phase and ambient gradient (grad(f))
        phase_val = self.phase_profile.get_phase(x, y)
        phi_x, phi_y = self.phase_profile.get_gradient(x, y)
        grad_f_x = phi_x
        grad_f_y = phi_y
        grad_f_z = be.zeros_like(phi_x)  # Since phi = phi(x, y)

        # 4. Compute surface gradient (G = grad(f) - (grad(f)·N)N)
        grad_f_dot_N = grad_f_x * nx + grad_f_y * ny  # grad_f_z is 0
        G_x = grad_f_x - grad_f_dot_N * nx
        G_y = grad_f_y - grad_f_dot_N * ny
        G_z = grad_f_z - grad_f_dot_N * nz  # This is just -grad_f_dot_N * nz

        # 5. Compute incident tangential component (k_in,‖ = k_in - (k_in·N)N)
        k_in_dot_N = k_ix * nx + k_iy * ny + k_iz * nz
        k_in_par_x = k_ix - k_in_dot_N * nx
        k_in_par_y = k_iy - k_in_dot_N * ny
        k_in_par_z = k_iz - k_in_dot_N * nz

        # 6. Compute outgoing tangential component (k_out,‖ = k_in,‖ + G)
        k_out_par_x = k_in_par_x + G_x
        k_out_par_y = k_in_par_y + G_y
        k_out_par_z = k_in_par_z + G_z

        # 7. Compute outgoing normal component (alpha)
        k_out_par_mag_sq = k_out_par_x**2 + k_out_par_y**2 + k_out_par_z**2
        k_out_mag_sq = (n2 * k0) ** 2
        R_sq = k_out_mag_sq - k_out_par_mag_sq  # (alpha)^2

        # 8. Handle TIR/Evanescence
        tir_mask = R_sq < 0.0
        rays.clip(tir_mask)
        R_sq = be.maximum(0.0, R_sq)
        alpha_mag = be.sqrt(R_sq)

        # 9. Choose sign for alpha (Normal component)
        # Refraction: points along +N
        # Reflection: points along -N
        alpha_sign = -1.0 if self.is_reflective else 1.0
        alpha = alpha_sign * alpha_mag

        # 10. Build full outgoing wavevector (k_out = k_out,‖ + alpha*N)
        k_out_x = k_out_par_x + alpha * nx
        k_out_y = k_out_par_y + alpha * ny
        k_out_z = k_out_par_z + alpha * nz

        # 11. Get new direction cosines
        k_out_mag = be.sqrt(k_out_x**2 + k_out_y**2 + k_out_z**2)
        l_o = k_out_x / k_out_mag
        m_o = k_out_y / k_out_mag
        n_o = k_out_z / k_out_mag

        # Update ray
        rays.L, rays.M, rays.N = l_o, m_o, n_o

        # Update OPD
        opd_shift = -phase_val / k0
        rays.opd = rays.opd + opd_shift

        # Apply coating/BSDF
        rays = self._apply_coating_and_bsdf(rays, nx, ny, nz)

        # Apply phase profile efficiency
        rays.i = rays.i * self.phase_profile.efficiency

        return rays

    def flip(self):
        """Flips the interaction model.

        For a phase profile defined on a plane, this operation does nothing.
        """
        pass

    def interact_paraxial_rays(self, rays: ParaxialRays) -> ParaxialRays:
        """Applies the phase gradient interaction to paraxial rays.

        Args:
            rays: The paraxial rays to interact with the surface.

        Returns:
            The interacted paraxial rays.
        """
        # Get incident state
        n1 = self.parent_surface.material_pre.n(rays.w)
        n2 = self.parent_surface.material_post.n(rays.w)
        k0 = 2 * be.pi / rays.w
        y = rays.y

        # Get paraxial gradient from the strategy
        paraxial_gradient = self.phase_profile.get_paraxial_gradient(y)

        # Apply geometric + gradient deflection
        grad_deflection = paraxial_gradient / k0

        if self.is_reflective:
            n = n1
            power = (
                -2 * n / self.parent_surface.geometry.radius
            )  # Will be zero for Plane
            u_geom = rays.u - y * power / n
            rays.u = u_geom + grad_deflection / n
        else:
            power = (n2 - n1) / self.parent_surface.geometry.radius
            u_geom = (n1 / n2) * rays.u - y * power / n2
            rays.u = u_geom + grad_deflection / n2

        return rays

    def to_dict(self) -> dict:
        """Serializes the phase profile to a dictionary.

        Returns:
            A dictionary representation of the interaction model.
        """
        data = super().to_dict()
        data.update({"phase_profile": self.phase_profile.to_dict()})
        return data

    @classmethod
    def from_dict(cls, data: dict, parent_surface: Surface) -> PhaseInteractionModel:
        """Deserializes an interaction model from a dictionary.

        Args:
            data: A dictionary representation of an interaction model.
            parent_surface: The surface to which this model is attached.

        Returns:
            An instance of a `PhaseInteractionModel`.
        """
        phase_profile = BasePhaseProfile.from_dict(data.pop("phase_profile"))
        data.pop("type", None)
        is_reflective = data.pop("is_reflective", False)
        return cls(
            parent_surface,
            phase_profile=phase_profile,
            is_reflective=is_reflective,
            **data,
        )
