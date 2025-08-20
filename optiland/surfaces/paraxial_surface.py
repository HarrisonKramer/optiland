"""Paraxial Surface

This module contains the class ParaxialSurface, which is a subclass of the
Surface class. It is used to represent a paraxial surface, or a thin lens,
which is defined simply by its effective focal length in air. This class is
used to model the behavior of a lens in the paraxial approximation and can be
used for first-order layout of optical systems.

Kramer Harrison, 2024
"""

from __future__ import annotations

import optiland.backend as be
from optiland.coatings import BaseCoating
from optiland.geometries import BaseGeometry
from optiland.materials import BaseMaterial
from optiland.physical_apertures import BaseAperture
from optiland.rays.polarized_rays import PolarizedRays
from optiland.scatter import BaseBSDF
from optiland.surfaces.standard_surface import Surface


class ParaxialSurface(Surface):
    """Paraxial Surface

    This class is used to represent a paraxial surface, which is planar, yet
    imparts a phase shift to the incident light, causing refraction. This class
    is used to model the behavior of a lens in the paraxial approximation and
    can be used for first-order layout of optical systems.
    """

    def __init__(
        self,
        focal_length,
        geometry,
        material_pre,
        material_post,
        is_stop=False,
        aperture=None,
        coating=None,
        bsdf=None,
        is_reflective=False,
        surface_type="paraxial",
        comment="",
    ):
        self.f = be.array(focal_length)
        super().__init__(
            geometry,
            material_pre,
            material_post,
            is_stop,
            aperture,
            coating,
            bsdf,
            is_reflective,
            surface_type,
            comment,
        )

    def _interact(self, rays):
        """Interacts the rays with the surface by either reflecting or refracting

        Note that phase is added assuming a thin lens as a phase
        transformation. A cosine correction is applied for rays propagating
        off-axis. This correction is equivalent to the ray z direction cosine.

        Args:
            rays: The rays.

        Returns:
            RealRays: The refracted rays.

        """
        # add optical path length - workaround for now
        # TODO: develop more robust method
        rays.opd = rays.opd - (rays.x**2 + rays.y**2) / (2 * self.f * rays.N)

        n1 = self.material_pre.n(rays.w)

        n2 = -n1 if self.is_reflective else self.material_post.n(rays.w)

        ux1 = rays.L / rays.N
        uy1 = rays.M / rays.N

        ux2 = 1 / n2 * (n1 * ux1 - rays.x / self.f)
        uy2 = 1 / n2 * (n1 * uy1 - rays.y / self.f)

        L = ux2
        M = uy2

        # only normalize if required
        if self.bsdf or self.coating or isinstance(rays, PolarizedRays):
            rays.normalize()

        # if there is a surface scatter model, modify ray properties
        if self.bsdf:
            rays = self.bsdf.scatter(rays, nx=0, ny=0, nz=1)

        # if there is a coating, modify ray properties
        if self.coating:
            rays = self.coating.interact(
                rays,
                reflect=self.is_reflective,
                nx=0,
                ny=0,
                nz=1,
            )
        else:
            # update polarization matrices, if PolarizedRays
            rays.update()

        # paraxial approximation -> direction is not necessarily unit vector
        rays.L = L
        rays.M = M
        rays.N = be.ones_like(L)
        rays.is_normalized = False

        return rays

    def _trace_paraxial(self, rays):
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

        n1 = self.material_pre.n(rays.w)
        if self.is_reflective:
            # reflect (derived from paraxial equations when n'=-n)
            rays.u = rays.y / (self.f * n1) - rays.u

        else:
            # surface power
            n2 = self.material_post.n(rays.w)

            # refract
            rays.u = 1 / n2 * (n1 * rays.u - rays.y / self.f)

        # inverse transform coordinate system
        self.geometry.globalize(rays)

        self._record(rays)

        return rays

    def to_dict(self):
        """Returns a dictionary representation of the surface."""
        return {
            "type": self.__class__.__name__,
            "focal_length": self.f,
            "geometry": self.geometry.to_dict(),
            "material_pre": self.material_pre.to_dict(),
            "material_post": self.material_post.to_dict(),
            "is_stop": self.is_stop,
            "aperture": self.aperture.to_dict() if self.aperture else None,
            "coating": self.coating.to_dict() if self.coating else None,
            "bsdf": self.bsdf.to_dict() if self.bsdf else None,
            "is_reflective": self.is_reflective,
            "comment": self.comment,
        }

    @classmethod
    def _from_dict(cls, data):
        """Protected deserialization logic for direct initialization.

        Args:
            data (dict): The dictionary representation of the surface.

        Returns:
            Surface: The surface.

        """
        focal_length = data["focal_length"]
        geometry = BaseGeometry.from_dict(data["geometry"])
        material_pre = BaseMaterial.from_dict(data["material_pre"])
        material_post = BaseMaterial.from_dict(data["material_post"])
        aperture = (
            BaseAperture.from_dict(data["aperture"]) if data["aperture"] else None
        )
        coating = BaseCoating.from_dict(data["coating"]) if data["coating"] else None
        bsdf = BaseBSDF.from_dict(data["bsdf"]) if data["bsdf"] else None

        return ParaxialSurface(
            focal_length,
            geometry,
            material_pre,
            material_post,
            data["is_stop"],
            aperture,
            coating,
            bsdf,
            data["is_reflective"],
            comment=data.get("comment", ""),
        )


class ParaxialToThickLensConverter:
    """
    Converts a ParaxialSurface into an equivalent thick lens composed of two
    real surfaces.

    Args:
        paraxial_surface: The ParaxialSurface to convert.
        optic: The parent Optic instance containing the paraxial surface.
        material: The lens material. Can be:
            - A string (e.g., "N-BK7", resolved via Material lookup).
            - A float (refractive index, creates an IdealMaterial).
            - A BaseMaterial instance.
        center_thickness: The desired center thickness of the thick lens.
    """

    def __init__(
        self,
        paraxial_surface: ParaxialSurface,
        optic,
        material: str | float | BaseMaterial = "N-BK7",
        center_thickness: float = 3.0,  # Default center thickness in mm
    ):
        if not isinstance(paraxial_surface, ParaxialSurface):
            raise TypeError("paraxial_surface must be an instance of ParaxialSurface.")

        self.paraxial_surface = paraxial_surface
        self.optic = optic
        self.original_focal_length = paraxial_surface.f
        self.center_thickness = center_thickness

        self._material_instance = self._resolve_material(material)

    def _resolve_material(
        self, material_input: str | float | BaseMaterial
    ) -> BaseMaterial:
        """Resolves the material input to a BaseMaterial instance."""
        from optiland.materials.ideal import IdealMaterial
        from optiland.materials.material import Material

        if isinstance(material_input, BaseMaterial):
            return material_input
        elif isinstance(material_input, str):
            try:
                return Material(material_input)
            except Exception as e:
                raise ValueError(
                    f"Could not resolve material string '{material_input}': {e}"
                ) from e
        elif isinstance(material_input, int | float):
            return IdealMaterial(n=float(material_input))
        else:
            raise TypeError(
                "Invalid material type. Must be BaseMaterial, str, or float."
            )

    def convert(self):
        """
        Performs the conversion from paraxial to thick lens.

        This method will:
        1. Calculate the front and back radii of the thick lens.
        2. Remove the original paraxial surface from the optic.
        3. Create and add two new surfaces to the optic.
        """
        r1, r2 = self._calculate_radii()

        # Store original index before removal
        original_index = self._get_paraxial_surface_index()
        if original_index is None:
            raise RuntimeError("Original paraxial surface not found in optic.")

        self._remove_paraxial_surface(original_index)
        self._add_surfaces(r1, r2, original_index)

    def _get_paraxial_surface_index(self):
        """Finds the index of the self.paraxial_surface in the optic's surface list."""
        for i, s in enumerate(self.optic.surface_group.surfaces):
            if s is self.paraxial_surface:
                return i
        return None

    def _calculate_radii(self):
        """
        Calculates the front (R1) and back (R2) radii of curvature for the
        thick lens using the Lensmaker's equation.

        P = (n_lens - n_medium) * (1/R1 - 1/R2 +
                                   (n_lens - n_medium)*d / (n_lens*R1*R2))
        where P = 1/f (power), n_lens is lens refractive index, n_medium is
        surrounding medium refractive index (assumed air, n_medium=1), and d is
        center thickness.

        For a target focal length f_target (self.original_focal_length),
        and assuming n_medium = 1 (air):
        1/f_target = (n - 1) * (1/R1 - 1/R2 + (n - 1)*d / (n*R1*R2))

        This method uses a biconvex lens for positive focal lengths and a biconcave
        lens for negative focal lengths.
        - biconvex: R1 > 0, R2 < 0. Assume R1 = -R2 for simplicity.
        - biconcave: R1 < 0, R2 > 0. Assume R1 = -R2.

        Returns:
            tuple[float, float]: (R1, R2)
        """
        n = self._material_instance.n(self.optic.primary_wavelength)
        if hasattr(n, "item"):  # If n is a 0-dim array/tensor
            n = n.item()
        f_target = self.original_focal_length
        d = self.center_thickness

        if abs(f_target) < 1e-9:
            return be.inf, be.inf

        P = 1.0 / f_target  # Power
        r1, r2 = 0.0, 0.0

        if f_target > 0:
            # Biconvex: P*n*R1^2 - 2*n*(n-1)*R1 + (n-1)^2*d = 0. For R1 = -R2.
            a_quad = P * n
            b_quad = -2 * n * (n - 1)
            c_quad = (n - 1) ** 2 * d

            if abs(a_quad) < 1e-9:
                if abs(b_quad) < 1e-9:
                    raise ValueError("Cannot solve for R1 in biconvex (P=0, n=1).")
                r1 = -c_quad / b_quad  # Linear case
            else:
                discriminant = b_quad**2 - 4 * a_quad * c_quad
                if discriminant < 0:
                    raise ValueError("Biconvex: discriminant < 0, cannot find real R1.")

                sol1 = (-b_quad + be.sqrt(discriminant)) / (2 * a_quad)
                sol2 = (-b_quad - be.sqrt(discriminant)) / (2 * a_quad)
                r1 = sol1 if sol1 > 0 else sol2
                if r1 <= 0:
                    r1 = sol2 if sol2 > 0 else sol1
                    if r1 <= 0:
                        raise ValueError("Biconvex: No positive R1 solution found.")
            r2 = -r1

        else:
            # Biconcave: P*n*R1^2 - 2*n*(n-1)*R1 + (n-1)^2*d = 0. For R1 = -R2.
            a_quad = P * n
            b_quad = -2 * n * (n - 1)
            c_quad = (n - 1) ** 2 * d
            if abs(a_quad) < 1e-9:
                if abs(b_quad) < 1e-9:
                    raise ValueError("Cannot solve for R1 in biconcave (P=0, n=1).")
                r1 = -c_quad / b_quad
            else:
                discriminant = b_quad**2 - 4 * a_quad * c_quad
                if discriminant < 0:
                    raise ValueError(
                        "Biconcave: discriminant < 0, cannot find real R1."
                    )
                # Choose solution for R1 < 0 if P < 0 (diverging)
                sol1 = (-b_quad + be.sqrt(discriminant)) / (2 * a_quad)
                sol2 = (-b_quad - be.sqrt(discriminant)) / (2 * a_quad)
                r1 = sol1 if sol1 < 0 else sol2
                if r1 >= 0:
                    r1 = sol2 if sol2 < 0 else sol1
                    if r1 >= 0:
                        raise ValueError("Biconcave: No negative R1 solution found.")
            r2 = -r1

        return float(r1), float(r2)

    def _add_surfaces(self, r1: float, r2: float, original_index: int):
        """
        Creates the two new standard Surface instances.
        Materials pre/post are set based on original paraxial surface context
        and the new lens material.
        """
        original_material_post = self.paraxial_surface.material_post

        # Surface 1: front surface of the thick lens
        self.optic.add_surface(
            index=original_index,
            radius=r1,
            material=self._material_instance,
            is_stop=self.paraxial_surface.is_stop,
            thickness=self.center_thickness,
            comment="Thick Lens - Surface 1",
        )

        # Surface 2: back surface of the thick lens
        self.optic.add_surface(
            index=original_index + 1,
            radius=r2,
            material=original_material_post,
            is_stop=False,  # Stop, if any, is on the first surface
            thickness=self.paraxial_surface.thickness,
            comment="Thick Lens - Surface 2",
        )

    def _remove_paraxial_surface(self, original_index: int):
        """
        Removes the original ParaxialSurface from the parent optic's
        surface_group using its index.
        """
        if not (0 < original_index < len(self.optic.surface_group.surfaces)):
            raise IndexError(
                f"Invalid index {original_index} for removing paraxial surface."
            )
        self.optic.surface_group.remove_surface(original_index)


def convert_to_thick_lens(lens):
    """
    Converts all paraxial surfaces in a lens into thick lenses

    Args:
        lens (Optic): the lens to be converted

    Returns:
        Optic: the converted lens
    """
    for surf in lens.surface_group.surfaces:
        if isinstance(surf, ParaxialSurface):
            converter = ParaxialToThickLensConverter(surf, lens)
            converter.convert()
    return lens
