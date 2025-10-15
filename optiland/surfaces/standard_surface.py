"""Standard Surface

This module defines the `Surface` class, which represents a surface in an
optical system. Surfaces are characterized by their geometry, materials before
and after the surface, and optional properties such as being an aperture stop,
having a physical aperture, and a coating. The module facilitates the tracing
of rays through these surfaces, accounting for refraction, reflection, and
absorption based on the surface properties and materials involved.

Kramer Harrison, 2023
"""

from __future__ import annotations

import optiland.backend as be
from optiland.coatings import BaseCoating, FresnelCoating
from optiland.geometries import BaseGeometry
from optiland.interactions.base import BaseInteractionModel
from optiland.interactions.refractive_reflective_model import RefractiveReflectiveModel
from optiland.materials import BaseMaterial
from optiland.physical_apertures import BaseAperture
from optiland.physical_apertures.radial import configure_aperture
from optiland.rays import BaseRays, ParaxialRays, RealRays
from optiland.scatter import BaseBSDF


class Surface:
    """Represents a standard refractice surface in an optical system.

    Args:
        geometry (BaseGeometry): The geometry of the surface.
        material_pre (BaseMaterial): The material before the surface.
        material_post (BaseMaterial): The material after the surface.
        is_stop (bool, optional): Indicates if the surface is the aperture
            stop. Defaults to False.
        aperture (BaseAperture, int, float, optional): The physical aperture of the
            surface. Defaults to None. If a scalar is provided, it specifies the
            diameter of the lens.
        surface_type (str, optional): The type of surface. Defaults to None.
        comment (str, optional): A comment for the surface. Defaults to ''.
        interaction_model (BaseInteractionModel, optional): The interaction
            model for the surface. Defaults to None.

    """

    _registry = {}  # registry for all surfaces

    def __init__(
        self,
        geometry: BaseGeometry,
        material_pre: BaseMaterial,
        material_post: BaseMaterial,
        is_stop: bool = False,
        aperture: BaseAperture = None,
        surface_type: str = None,
        comment: str = "",
        interaction_model: BaseInteractionModel = None,
    ):
        self.geometry = geometry
        self.material_pre = material_pre
        self.material_post = material_post
        self.is_stop = is_stop
        self.aperture = configure_aperture(aperture)
        self.semi_aperture = None
        self.surface_type = surface_type
        self.comment = comment

        if interaction_model is None:
            self.interaction_model = RefractiveReflectiveModel(
                geometry=self.geometry,
                material_pre=self.material_pre,
                material_post=self.material_post,
                is_reflective=False,
                coating=None,
                bsdf=None,
            )
        else:
            self.interaction_model = interaction_model

        self.thickness = 0.0  # used for surface positioning

        self.reset()

    def flip(self):
        """Flips the surface, swapping materials and reversing geometry."""
        self.material_pre, self.material_post = self.material_post, self.material_pre
        self.geometry.flip()

        # Re-create the interaction model with flipped properties
        self.interaction_model.flip()

        if isinstance(self.interaction_model.coating, FresnelCoating):
            self.set_fresnel_coating()
        elif self.interaction_model.coating is not None and hasattr(
            self.interaction_model.coating, "flip"
        ):
            self.interaction_model.coating.flip()

        self.reset()

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        Surface._registry[cls.__name__] = cls

    def trace(self, rays: BaseRays):
        """Traces the given rays through the surface.

        Args:
            rays (BaseRays): The rays to be traced.

        Returns:
            BaseRays: The traced rays.

        """
        # reset recorded information
        self.reset()

        # transform coordinate system
        self.geometry.localize(rays)

        if isinstance(rays, ParaxialRays):
            # propagate to this surface
            t = -rays.z
            rays.propagate(t)

            # interact with surface
            rays = self.interaction_model.interact_paraxial_rays(rays)

        elif isinstance(rays, RealRays):
            # find distance from rays to the surface
            t = self.geometry.distance(rays)

            # propagate the rays a distance t through material
            self.material_pre.propagation_model.propagate(rays, t)

            # update OPD
            rays.opd = rays.opd + be.abs(t * self.material_pre.n(rays.w))

            # if there is a limiting aperture, clip rays outside of it
            if self.aperture:
                self.aperture.clip(rays)

            # interact with surface
            rays = self.interaction_model.interact_real_rays(rays)

        # inverse transform coordinate system
        self.geometry.globalize(rays)

        # record ray information
        self._record(rays)

        return rays

    def set_semi_aperture(self, r_max: float):
        """Sets the physical semi-aperture of the surface.

        Args:
            r_max (float): The maximum radius of the semi-aperture.

        """
        self.semi_aperture = r_max

    def reset(self):
        """Resets the recorded information of the surface."""
        self.y = be.empty(0)
        self.u = be.empty(0)
        self.x = be.empty(0)
        self.y = be.empty(0)
        self.z = be.empty(0)

        self.L = be.empty(0)
        self.M = be.empty(0)
        self.N = be.empty(0)

        self.intensity = be.empty(0)
        self.aoi = be.empty(0)
        self.opd = be.empty(0)

    def set_fresnel_coating(self):
        """Sets the coating of the surface to a Fresnel coating."""
        self.coating = FresnelCoating(self.material_pre, self.material_post)
        self.interaction_model.coating = self.coating

    def _record(self, rays):
        """Records the ray information.

        Args:
            rays: The rays.

        """
        if isinstance(rays, ParaxialRays):
            self.y = be.copy(be.atleast_1d(rays.y))
            self.u = be.copy(be.atleast_1d(rays.u))
        elif isinstance(rays, RealRays):
            self.x = be.copy(be.atleast_1d(rays.x))
            self.y = be.copy(be.atleast_1d(rays.y))
            self.z = be.copy(be.atleast_1d(rays.z))

            self.L = be.copy(be.atleast_1d(rays.L))
            self.M = be.copy(be.atleast_1d(rays.M))
            self.N = be.copy(be.atleast_1d(rays.N))

            self.intensity = be.copy(be.atleast_1d(rays.i))
            self.opd = be.copy(be.atleast_1d(rays.opd))

    def is_rotationally_symmetric(self):
        """Returns True if the surface is rotationally symmetric, False otherwise."""
        if not self.geometry.is_symmetric:
            return False

        cs = self.geometry.cs
        return not (cs.rx != 0 or cs.ry != 0 or cs.x != 0 or cs.y != 0)

    def to_dict(self):
        """Returns a dictionary representation of the surface."""
        # backward compatibility
        if not hasattr(self, "interaction_model"):
            self.interaction_model = RefractiveReflectiveModel(
                geometry=self.geometry,
                material_pre=self.material_pre,
                material_post=self.material_post,
                is_reflective=self.is_reflective,
                coating=self.coating,
                bsdf=self.bsdf,
            )

        return {
            "type": self.__class__.__name__,
            "thickness": self.thickness,
            "geometry": self.geometry.to_dict(),
            "material_pre": self.material_pre.to_dict(),
            "material_post": self.material_post.to_dict(),
            "is_stop": self.is_stop,
            "aperture": self.aperture.to_dict() if self.aperture else None,
            "interaction_model": self.interaction_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        """Creates a surface from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the surface.

        Returns:
            Surface: The surface.

        """
        if "type" not in data:
            raise ValueError("Missing 'type' field.")

        type_name = data["type"]
        subclass = cls._registry.get(type_name, cls)
        return subclass._from_dict(data)

    @classmethod
    def _from_dict(cls, data):
        """Protected deserialization logic for direct initialization.

        Args:
            data (dict): The dictionary representation of the surface.

        Returns:
            Surface: The surface.

        """
        surface_type = data.get("type")
        geometry = BaseGeometry.from_dict(data["geometry"])
        material_pre = BaseMaterial.from_dict(data["material_pre"])
        material_post = BaseMaterial.from_dict(data["material_post"])
        aperture = (
            BaseAperture.from_dict(data["aperture"]) if data.get("aperture") else None
        )

        interaction_model_data = data.get("interaction_model")
        if interaction_model_data:
            interaction_model = BaseInteractionModel.from_dict(
                interaction_model_data, geometry, material_pre, material_post
            )
        else:
            # Backward compatibility
            coating = (
                BaseCoating.from_dict(data["coating"]) if data.get("coating") else None
            )
            bsdf = BaseBSDF.from_dict(data["bsdf"]) if data.get("bsdf") else None
            interaction_model = RefractiveReflectiveModel(
                geometry=geometry,
                material_pre=material_pre,
                material_post=material_post,
                is_reflective=data.get("is_reflective", False),
                coating=coating,
                bsdf=bsdf,
            )

        surface_class = cls._registry.get(surface_type, cls)
        surface = surface_class(
            geometry=geometry,
            material_pre=material_pre,
            material_post=material_post,
            is_stop=data.get("is_stop", False),
            aperture=aperture,
            comment=data.get("comment", ""),
            interaction_model=interaction_model,
        )
        surface.thickness = data.get("thickness", 0.0)
        return surface
