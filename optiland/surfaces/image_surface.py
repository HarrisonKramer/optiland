"""Image Surface

This module contains the ImageSurface class, which represents an image surface
in an optical system.

Kramer Harrison, 2024
"""

from optiland.surfaces.standard_surface import Surface
from optiland.rays import ParaxialRays
from optiland.materials import BaseMaterial
from optiland.physical_apertures import BaseAperture
from optiland.geometries import BaseGeometry


class ImageSurface(Surface):
    """
    Represents an image surface in an optical system.

    Args:
        geometry (BaseGeometry): The geometry of the surface.
        material_pre (BaseMaterial): The material before the surface.
        aperture (BaseAperture, optional): The aperture of the surface.
            Defaults to None.
    """

    def __init__(self, geometry: BaseGeometry, material_pre: BaseMaterial,
                 material_post: BaseMaterial, aperture: BaseAperture = None):
        super().__init__(
            geometry=geometry,
            material_pre=material_pre,
            material_post=material_post,
            is_stop=False,
            aperture=aperture
        )

    def _trace_paraxial(self, rays: ParaxialRays):
        """
        Traces paraxial rays through the surface.

        Args:
            rays (ParaxialRays): The paraxial rays to be traced.
        """
        # reset recorded information
        self.reset()

        # transform coordinate system
        self.geometry.localize(rays)

        # propagate to this surface
        t = -rays.z
        rays.propagate(t)

        # inverse transform coordinate system
        self.geometry.globalize(rays)

        self._record(rays)

        return rays

    def _interact(self, rays):
        """
        Interacts rays with the surface.

        Args:
            rays: The rays to be interacted with the surface.

        Returns:
            RealRays: The modified rays after interaction with the surface.
        """
        return rays
