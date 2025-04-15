"""Object Surface

This module contains the ObjectSurface class, which represents an object
surface in an optical system.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.geometries import BaseGeometry
from optiland.materials import BaseMaterial
from optiland.rays import ParaxialRays, RealRays
from optiland.surfaces.standard_surface import Surface


class ObjectSurface(Surface):
    """Represents an object surface in an optical system.

    Args:
        geometry (Geometry): The geometry of the surface.
        material_post (Material): The material of the surface after
            interaction.
        comment (str, optional): A comment for the surface. Defaults
            to ''.

    Attributes:
        is_infinite (bool): Indicates whether the surface is infinitely
            far away.

    """

    def __init__(self, geometry, material_post, comment=""):
        super().__init__(
            geometry=geometry,
            material_pre=material_post,
            material_post=material_post,
            is_stop=False,
            aperture=None,
            comment=comment,
        )

    @property
    def is_infinite(self):
        """Returns True if the surface is infinitely far away, False otherwise."""
        return be.isinf(self.geometry.cs.z)

    def set_aperture(self):
        """Sets the aperture of the surface."""

    def trace(self, rays):
        """Traces the given rays through the surface.

        Args:
            rays (Rays): The rays to be traced.

        Returns:
            RealRays: The traced rays.

        """
        # reset recorded information
        self.reset()

        # record ray information
        self._record(rays)

        return rays

    def _trace_paraxial(self, rays: ParaxialRays):
        """Traces the given paraxial rays through the surface.

        Args:
            rays (ParaxialRays): The paraxial rays to be traced.

        """

    def _trace_real(self, rays: RealRays):
        """Traces the given real rays through the surface.

        Args:
            rays (RealRays): The real rays to be traced.

        """

    def _interact(self, rays):
        """Interacts the given rays with the surface.

        Args:
            rays (Rays): The rays to be interacted.

        Returns:
            RealRays: The interacted rays.

        """
        return rays

    def to_dict(self):
        """Returns a dictionary representation of the surface."""
        return {
            "type": self.__class__.__name__,
            "geometry": self.geometry.to_dict(),
            "material_post": self.material_post.to_dict(),
        }

    @classmethod
    def _from_dict(cls, data):
        """Creates a surface from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the surface.

        Returns:
            Surface: The surface.

        """
        geometry = BaseGeometry.from_dict(data["geometry"])
        material_post = BaseMaterial.from_dict(data["material_post"])
        return cls(geometry, material_post)
