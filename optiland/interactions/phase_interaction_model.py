"""
Provides the phase interaction model.
"""

from __future__ import annotations

import typing

from optiland import backend as be
from optiland.geometries.plane import Plane
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
        self, parent_surface: Surface, phase_profile: BasePhaseProfile, **kwargs
    ):
        super().__init__(parent_surface, **kwargs)
        self.phase_profile = phase_profile

    def interact_real_rays(self, rays: RealRays) -> RealRays:
        if not isinstance(self.parent_surface.geometry, Plane):
            raise TypeError(
                "PhaseInteractionModel can only be used with a Plane geometry."
            )
        """Applies the phase gradient interaction to real rays.

        Args:
            rays: The real rays to interact with the surface.

        Returns:
            The interacted real rays.
        """
        # Get incident state
        x, y = rays.x, rays.y
        l_i, m_i, n_i = rays.L, rays.M, rays.N
        rays.L0, rays.M0, rays.N0 = l_i, m_i, n_i
        n1 = self.parent_surface.material_pre.n(rays.w)
        n2 = self.parent_surface.material_post.n(rays.w)
        if self.is_reflective:
            n2 = n1
        k0 = 2 * be.pi / rays.w

        # Get phase and gradient from the strategy
        phase_val = self.phase_profile.get_phase(x, y)
        phi_x, phi_y = self.phase_profile.get_gradient(x, y)

        # Apply Generalized Snell's Law
        k_ix = n1 * k0 * l_i
        k_iy = n1 * k0 * m_i
        k_ox = k_ix + phi_x
        k_oy = k_iy + phi_y

        # Compute outgoing direction cosines
        l_o = k_ox / (n2 * k0)
        m_o = k_oy / (n2 * k0)

        # Compute outgoing z-component and handle TIR
        tmp = 1.0 - l_o**2 - m_o**2
        tir_mask = tmp < 0.0
        rays.clip(tir_mask)
        p_o_mag = be.sqrt(be.maximum(0.0, tmp))

        # Handle direction
        sign = -be.sign(n_i) if self.is_reflective else be.sign(n_i)
        n_o = sign * p_o_mag

        # Update ray
        rays.L, rays.M, rays.N = l_o, m_o, n_o

        # Update OPD
        opd_shift = phase_val / k0
        rays.opd = rays.opd + opd_shift

        # Apply coating/BSDF
        nx, ny, nz = self.parent_surface.geometry.surface_normal(rays)
        rays = self._apply_coating_and_bsdf(rays, nx, ny, nz)
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
            rays.u = u_geom - grad_deflection / n2

        return rays

    def to_dict(self) -> dict:
        """Serializes the interaction model to a dictionary.

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
        data.pop("type", None) # Remove type key to avoid passing it to constructor
        return cls(parent_surface, phase_profile=phase_profile, **data)
