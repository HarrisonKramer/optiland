import numpy as np
from optiland.rays import RealRays, ParaxialRays
from optiland.surfaces.standard_surface import Surface


class ObjectSurface(Surface):
    """
    Represents an object surface in an optical system.

    Args:
        geometry (Geometry): The geometry of the surface.
        material_post (Material): The material of the surface after
            interaction.

    Attributes:
        is_infinite (bool): Indicates whether the surface is infinitely
            far away.
    """

    def __init__(self, geometry, material_post):
        super().__init__(
            geometry=geometry,
            material_pre=material_post,
            material_post=material_post,
            is_stop=False,
            aperture=None
        )

    @property
    def is_infinite(self):
        """
        Returns True if the surface is infinitely far away, False otherwise.
        """
        return np.isinf(self.geometry.cs.z)

    def set_aperture(self):
        """
        Sets the aperture of the surface.
        """
        pass

    def trace(self, rays):
        """
        Traces the given rays through the surface.

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
        """
        Traces the given paraxial rays through the surface.

        Args:
            rays (ParaxialRays): The paraxial rays to be traced.
        """
        pass

    def _trace_real(self, rays: RealRays):
        """
        Traces the given real rays through the surface.

        Args:
            rays (RealRays): The real rays to be traced.
        """
        pass

    def _interact(self, rays):
        """
        Interacts the given rays with the surface.

        Args:
            rays (Rays): The rays to be interacted.

        Returns:
            RealRays: The interacted rays.
        """
        return rays
