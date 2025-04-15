"""Optic Updater Module

This module contains the OpticModifier class, which is responsible for updating the
optical system properties, such as the surface radii of curvature, thicknesses,
materials, conic constants, polarization, etc.

Kramer Harrison, 2025
"""

from typing import Union

import optiland.backend as be
from optiland.geometries import Plane, StandardGeometry
from optiland.materials import IdealMaterial
from optiland.rays import PolarizationState


class OpticUpdater:
    """Class to update or modify an optical system

    This class is responsible for updating the optical system properties, such as
    the surface radii of curvature, thicknesses, materials, conic constants,
    polarization, etc.

    Args:
        optic (Optic): The optical system to be modified.
    """

    def __init__(self, optic):
        self.optic = optic

    def set_radius(self, value, surface_number):
        """Set the radius of curvature of a surface.

        Args:
            value (float): The value of the radius.
            surface_number (int): The index of the surface.

        """
        surface = self.optic.surface_group.surfaces[surface_number]

        # change geometry from plane to standard
        if isinstance(surface.geometry, Plane):
            cs = surface.geometry.cs
            new_geometry = StandardGeometry(cs, radius=value, conic=0)
            surface.geometry = new_geometry
        else:
            surface.geometry.radius = value

    def set_conic(self, value, surface_number):
        """Set the conic constant of a surface.

        Args:
            value (float): The value of the conic constant.
            surface_number (int): The index of the surface.

        """
        surface = self.optic.surface_group.surfaces[surface_number]
        surface.geometry.k = value

    def set_thickness(self, value, surface_number):
        """Set the thickness of a surface.

        Args:
            value (float): The value of the thickness.
            surface_number (int): The index of the surface.

        """
        positions = self.optic.surface_group.positions
        delta_t = value - positions[surface_number + 1] + positions[surface_number]
        positions[surface_number + 1 :] += delta_t
        positions -= positions[1]  # force surface 1 to be at zero
        for k, surface in enumerate(self.optic.surface_group.surfaces):
            surface.geometry.cs.z = float(positions[k])

    def set_index(self, value, surface_number):
        """Set the index of refraction of a surface.

        Args:
            value (float): The value of the index of refraction.
            surface_number (int): The index of the surface.

        """
        surface = self.optic.surface_group.surfaces[surface_number]
        new_material = IdealMaterial(n=value, k=0)
        surface.material_post = new_material

        surface_post = self.optic.surface_group.surfaces[surface_number + 1]
        surface_post.material_pre = new_material

    def set_asphere_coeff(self, value, surface_number, aspher_coeff_idx):
        """Set the asphere coefficient on a surface

        Args:
            value (float): The value of aspheric coefficient
            surface_number (int): The index of the surface.
            aspher_coeff_idx (int): index of the aspheric coefficient on the
                surface

        """
        surface = self.optic.surface_group.surfaces[surface_number]
        surface.geometry.c[aspher_coeff_idx] = value

    def set_polarization(self, polarization: Union[PolarizationState, str]):
        """Set the polarization state of the optic.

        Args:
            polarization (Union[PolarizationState, str]): The polarization
                state to set. It can be either a `PolarizationState` object or
                'ignore'.

        """
        if isinstance(polarization, str) and polarization != "ignore":
            raise ValueError(
                "Invalid polarization state. Must be either "
                'PolarizationState or "ignore".',
            )
        self.optic.polarization = polarization

    def scale_system(self, scale_factor):
        """Scales the optical system by a given scale factor.

        Args:
            scale_factor (float): The factor by which to scale the system.

        """
        num_surfaces = self.optic.surface_group.num_surfaces
        radii = self.optic.surface_group.radii
        thicknesses = [
            self.optic.surface_group.get_thickness(surf_idx)[0]
            for surf_idx in range(num_surfaces - 1)
        ]

        # Scale radii & thicknesses
        for surf_idx in range(num_surfaces):
            if not be.isinf(radii[surf_idx]):
                self.set_radius(radii[surf_idx] * scale_factor, surf_idx)

            if surf_idx != num_surfaces - 1 and not be.isinf(thicknesses[surf_idx]):
                self.set_thickness(thicknesses[surf_idx] * scale_factor, surf_idx)

        # Scale aperture, if aperture type is EPD
        if self.optic.aperture.ap_type == "EPD":
            self.optic.aperture.value *= scale_factor

        # Scale physical apertures
        for surface in self.optic.surface_group.surfaces:
            if surface.aperture is not None:
                surface.aperture.scale(scale_factor)

    def update_paraxial(self):
        """Update the semi-aperture of the surfaces based on the paraxial
        analysis.
        """
        ya, _ = self.optic.paraxial.marginal_ray()
        yb, _ = self.optic.paraxial.chief_ray()
        ya = be.abs(be.ravel(ya))
        yb = be.abs(be.ravel(yb))
        for k, surface in enumerate(self.optic.surface_group.surfaces):
            surface.set_semi_aperture(r_max=ya[k] + yb[k])
            self.update_normalization(surface)

    def update_normalization(self, surface) -> None:
        """Update the normalization radius of non-spherical surfaces."""
        if surface.surface_type in [
            "even_asphere",
            "odd_asphere",
            "polynomial",
            "chebyshev",
        ]:
            surface.geometry.norm_x = surface.semi_aperture * 1.1
            surface.geometry.norm_y = surface.semi_aperture * 1.1
        if surface.surface_type == "zernike":
            surface.geometry.norm_radius = surface.semi_aperture * 1.1

    def update(self) -> None:
        """Update the surface properties (pickups, solves, paraxial properties)."""
        self.optic.pickups.apply()
        self.optic.solves.apply()

        if any(
            surface.surface_type in ["chebyshev", "zernike"]
            for surface in self.optic.surface_group.surfaces
        ):
            self.update_paraxial()

    def image_solve(self):
        """Update the image position such that the marginal ray crosses the
        optical axis at the image location.
        """
        ya, ua = self.optic.paraxial.marginal_ray()
        offset = float(ya[-1, 0] / ua[-1, 0])
        self.optic.surface_group.surfaces[-1].geometry.cs.z -= offset
