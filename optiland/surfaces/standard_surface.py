"""Standard Surface

This module defines the `Surface` class, which represents a surface in an
optical system. Surfaces are characterized by their geometry, materials before
and after the surface, and optional properties such as being an aperture stop,
having a physical aperture, and a coating. The module facilitates the tracing
of rays through these surfaces, accounting for refraction, reflection, and
absorption based on the surface properties and materials involved.

Kramer Harrison, 2023
"""

import optiland.backend as be
from optiland.coatings import BaseCoating, FresnelCoating
from optiland.geometries import BaseGeometry
from optiland.materials import BaseMaterial
from optiland.physical_apertures import BaseAperture
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
        aperture (BaseAperture, optional): The physical aperture of the
            surface. Defaults to None.
        coating (BaseCoating, optional): The coating applied to the surface.
            Defaults to None.
        comment (str, optional): A comment for the surface. Defaults to ''.

    """

    _registry = {}  # registry for all surfaces

    def __init__(
        self,
        geometry: BaseGeometry,
        material_pre: BaseMaterial,
        material_post: BaseMaterial,
        is_stop: bool = False,
        aperture: BaseAperture = None,
        coating: BaseCoating = None,
        bsdf: BaseBSDF = None,
        is_reflective: bool = False,
        surface_type: str = None,
        comment: str = "",
    ):
        self.geometry = geometry
        self.material_pre = material_pre
        self.material_post = material_post
        self.is_stop = is_stop
        self.aperture = aperture
        self.semi_aperture = None
        self.coating = coating
        self.bsdf = bsdf
        self.is_reflective = is_reflective
        self.surface_type = surface_type
        self.comment = comment

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
        if isinstance(rays, ParaxialRays):
            return self._trace_paraxial(rays)
        if isinstance(rays, RealRays):
            return self._trace_real(rays)

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

    def _interact(self, rays):
        """Interacts the rays with the surface by either reflecting or refracting

        Args:
            rays: The rays.

        Returns:
            RealRays: The refracted rays.

        """
        # find surface normals
        nx, ny, nz = self.geometry.surface_normal(rays)

        # Interact with surface (refract or reflect)
        if self.is_reflective:
            rays.reflect(nx, ny, nz)
        else:
            n1 = self.material_pre.n(rays.w)
            n2 = self.material_post.n(rays.w)
            rays.refract(nx, ny, nz, n1, n2)

        # if there is a surface scatter model, modify ray properties
        if self.bsdf:
            rays = self.bsdf.scatter(rays, nx, ny, nz)

        # if there is a coating, modify ray properties
        if self.coating:
            rays = self.coating.interact(
                rays,
                reflect=self.is_reflective,
                nx=nx,
                ny=ny,
                nz=nz,
            )
        else:
            # update polarization matrices, if PolarizedRays
            rays.update()

        return rays

    def _trace_paraxial(self, rays: ParaxialRays):
        """Traces paraxial rays through the surface.

        Args:
            ParaxialRays: The paraxial rays to be traced.

        """
        # reset recorded information
        self.reset()

        # transform coordinate system
        self.geometry.localize(rays)

        # propagate to this surface
        t = -rays.z
        rays.propagate(t)

        if self.is_reflective:
            # reflect (derived from paraxial equations when n'=-n)
            rays.u = -rays.u - 2 * rays.y / self.geometry.radius

        else:
            # surface power
            n1 = self.material_pre.n(rays.w)
            n2 = self.material_post.n(rays.w)
            power = (n2 - n1) / self.geometry.radius

            # refract
            rays.u = 1 / n2 * (n1 * rays.u - rays.y * power)

        # inverse transform coordinate system
        self.geometry.globalize(rays)

        self._record(rays)

        return rays

    def _trace_real(self, rays: RealRays):
        """Traces real rays through the surface.

        Args:
            rays (RealRays): The real rays to be traced.

        Returns:
            RealRays: The traced real rays.

        """
        # reset recorded information
        self.reset()

        # transform coordinate system
        self.geometry.localize(rays)

        # find distance from rays to the surface
        t = self.geometry.distance(rays)

        # propagate the rays a distance t through material
        rays.propagate(t, self.material_pre)

        # update OPD
        rays.opd += be.abs(t * self.material_pre.n(rays.w))

        # if there is a limiting aperture, clip rays outside of it
        if self.aperture:
            self.aperture.clip(rays)

        # interact with surface
        rays = self._interact(rays)

        # inverse transform coordinate system
        self.geometry.globalize(rays)

        # record ray information
        self._record(rays)

        return rays

    def is_rotationally_symmetric(self):
        """Returns True if the surface is rotationally symmetric, False otherwise."""
        if not self.geometry.is_symmetric:
            return False

        cs = self.geometry.cs
        return not (cs.rx != 0 or cs.ry != 0 or cs.x != 0 or cs.y != 0)

    def to_dict(self):
        """Returns a dictionary representation of the surface."""
        return {
            "type": self.__class__.__name__,
            "geometry": self.geometry.to_dict(),
            "material_pre": self.material_pre.to_dict(),
            "material_post": self.material_post.to_dict(),
            "is_stop": self.is_stop,
            "aperture": self.aperture.to_dict() if self.aperture else None,
            "coating": self.coating.to_dict() if self.coating else None,
            "bsdf": self.bsdf.to_dict() if self.bsdf else None,
            "is_reflective": self.is_reflective,
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
            BaseAperture.from_dict(data["aperture"]) if data["aperture"] else None
        )
        coating = BaseCoating.from_dict(data["coating"]) if data["coating"] else None
        bsdf = BaseBSDF.from_dict(data["bsdf"]) if data["bsdf"] else None

        surface_class = cls._registry.get(surface_type, cls)

        return surface_class(
            geometry,
            material_pre,
            material_post,
            data["is_stop"],
            aperture,
            coating,
            bsdf,
            data["is_reflective"],
        )
