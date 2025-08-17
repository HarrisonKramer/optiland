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
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import BaseGeometry
from optiland.geometries.standard import StandardGeometry
from optiland.materials import BaseMaterial
from optiland.physical_apertures import BaseAperture
from optiland.rays.polarized_rays import PolarizedRays
from optiland.scatter import BaseBSDF
from optiland.surfaces.standard_surface import Surface  # Corrected import


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
    """

    def __init__(
        self,
        paraxial_surface: ParaxialSurface,
        optic,  # Optic type hint will be added once Optic is imported
        material: str | float | BaseMaterial = "N-BK7",
        center_thickness: float = 3.0,  # Default center thickness in mm
        lens_shape: str = "biconvex",
        # alignment: str = "center" # TODO: Implement different alignment strategies
    ):
        """
        Initializes the converter.

        Args:
            paraxial_surface: The ParaxialSurface to convert.
            optic: The parent Optic instance containing the paraxial surface.
            material: The lens material. Can be:
                - A string (e.g., "N-BK7", resolved via Material lookup).
                - A float (refractive index, creates an IdealMaterial).
                - A BaseMaterial instance.
            center_thickness: The desired center thickness of the thick lens.
            lens_shape: The shape of the lens ("biconvex", "plano-convex",
                          "convex-plano", "biconcave", "plano-concave",
                          "concave-plano", "meniscus-convex", "meniscus-concave").
                          For meniscus lenses, R1 is the more curved surface.
            # alignment: How to align the thick lens relative to the original
            #              paraxial surface position. "center" aligns geometric
            #              center, "front" aligns front vertex, "back" aligns
            #              back vertex.
        """
        if not isinstance(paraxial_surface, ParaxialSurface):
            raise TypeError("paraxial_surface must be an instance of ParaxialSurface.")
        # Optic type hint will be added once Optic is imported
        # if not isinstance(optic, Optic):
        #     raise TypeError("optic must be an instance of Optic.")

        self.paraxial_surface = paraxial_surface
        self.optic = optic
        self.original_focal_length = paraxial_surface.f
        self.center_thickness = center_thickness
        self.lens_shape = lens_shape.lower()
        # self.alignment = alignment.lower() # TODO

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
        2. Create two new standard surfaces.
        3. Remove the original paraxial surface from the optic.
        4. Insert the two new surfaces into the optic.
        """
        r1, r2 = self._calculate_radii()
        surface1, surface2 = self._create_surfaces(r1, r2)

        # Store original index before removal
        original_index = self._get_paraxial_surface_index()
        if original_index is None:
            raise RuntimeError("Original paraxial surface not found in optic.")

        self._remove_paraxial_surface(original_index)
        self._insert_new_surfaces(surface1, surface2, original_index)

        # Update the optic to reflect changes (e.g., recalculate paraxial properties)
        if hasattr(self.optic, "update"):
            self.optic.update()

        return surface1, surface2

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

        This method needs to account for self.lens_shape.
        - "biconvex": R1 > 0, R2 < 0. Assume R1 = -R2 for simplicity.
        - "plano-convex": R1 = inf, R2 < 0 (or R1 > 0, R2 = inf).
        - "convex-plano": R1 > 0, R2 = inf.
        - "biconcave": R1 < 0, R2 > 0. Assume R1 = -R2.
        - "plano-concave": R1 = inf, R2 > 0.
        - "concave-plano": R1 < 0, R2 = inf.
        - "meniscus-convex": R1, R2 same sign, lens thicker in middle.
        - "meniscus-concave": R1, R2 same sign, lens thinner in middle.

        For biconvex/biconcave, assumes symmetric radii (R1 = -R2).
        For plano-convex/concave, one radius is infinity.
        Meniscus lenses require an additional parameter.

        Returns:
            tuple[float, float]: (R1, R2)
        """
        n = self._material_instance.n(self.optic.primary_wavelength)
        if hasattr(n, "item"):  # If n is a 0-dim array/tensor
            n = n.item()
        f_target = self.original_focal_length
        d = self.center_thickness

        if abs(f_target) < 1e-9:  # Effectively infinite focal length
            # For an afocal system, typically R1 = R2 or both plano.
            # This is ambiguous without more constraints for a single thick lens.
            # Returning two plano surfaces for afocal case.
            if self.lens_shape in [
                "biconvex",
                "biconcave",
                "meniscus-convex",
                "meniscus-concave",
                "plano-convex",
                "convex-plano",
                "plano-concave",
                "concave-plano",
            ]:
                return be.inf, be.inf
            else:
                raise ValueError(
                    f"Unsupported lens_shape '{self.lens_shape}' for afocal conversion."
                )

        P = 1.0 / f_target  # Power
        r1, r2 = 0.0, 0.0

        if self.lens_shape == "biconvex":
            # P*n*R1^2 - 2*n*(n-1)*R1 + (n-1)^2*d = 0. For R1 = -R2.
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
                # Choose solution for R1 > 0 if P > 0 (converging)
                # or R1 > 0 if P < 0 (diverging, unusual for biconvex name)
                sol1 = (-b_quad + be.sqrt(discriminant)) / (2 * a_quad)
                sol2 = (-b_quad - be.sqrt(discriminant)) / (2 * a_quad)
                r1 = sol1 if sol1 > 0 else sol2
                if r1 <= 0:  # If preferred sol isn't >0, try other, or raise
                    r1 = sol2 if sol2 > 0 else sol1
                    if r1 <= 0:
                        raise ValueError("Biconvex: No positive R1 solution found.")
            r2 = -r1

        elif self.lens_shape == "plano-convex":
            # R1 = inf. P = (n-1) * (-1/R2). R2 = -(n-1)/P
            r1 = be.inf
            r2 = -(n - 1) / P
            if f_target > 0 and r2 >= 0:  # Converging, R2 should be < 0
                raise ValueError(f"Plano-convex converging error: R2={r2} not < 0.")
            if f_target < 0 and r2 <= 0:  # Diverging, R2 should be > 0
                raise ValueError(f"Plano-convex diverging error: R2={r2} not > 0.")

        elif self.lens_shape == "convex-plano":
            # R2 = inf. P = (n-1) * (1/R1). R1 = (n-1)/P
            r2 = be.inf
            r1 = (n - 1) / P
            if f_target > 0 and r1 <= 0:  # Converging, R1 should be > 0
                raise ValueError(f"Convex-plano converging error: R1={r1} not > 0.")
            if f_target < 0 and r1 >= 0:  # Diverging, R1 should be < 0
                raise ValueError(f"Convex-plano diverging error: R1={r1} not < 0.")

        elif self.lens_shape == "biconcave":
            # P*n*R1^2 - 2*n*(n-1)*R1 + (n-1)^2*d = 0. For R1 = -R2.
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
                if r1 >= 0:  # If preferred sol isn't <0, try other, or raise
                    r1 = sol2 if sol2 < 0 else sol1
                    if r1 >= 0:
                        raise ValueError("Biconcave: No negative R1 solution found.")
            r2 = -r1

        elif self.lens_shape == "plano-concave":
            # R1 = inf. P = (n-1) * (-1/R2). R2 = -(n-1)/P
            r1 = be.inf
            r2 = -(n - 1) / P
            if f_target < 0 and r2 <= 0:  # Diverging, R2 should be > 0
                raise ValueError(f"Plano-concave diverging error: R2={r2} not > 0.")
            if f_target > 0 and r2 >= 0:  # Converging, R2 should be < 0
                raise ValueError(f"Plano-concave converging error: R2={r2} not < 0.")

        elif self.lens_shape == "concave-plano":
            # R2 = inf. P = (n-1) * (1/R1). R1 = (n-1)/P
            r2 = be.inf
            r1 = (n - 1) / P
            if f_target < 0 and r1 >= 0:  # Diverging, R1 should be < 0
                raise ValueError(f"Concave-plano diverging error: R1={r1} not < 0.")
            if f_target > 0 and r1 <= 0:  # Converging, R1 should be > 0
                raise ValueError(f"Concave-plano converging error: R1={r1} not > 0.")

        elif "meniscus" in self.lens_shape:
            raise NotImplementedError(
                "Meniscus lens shape requires an additional parameter "
                "(e.g., one radius or shape factor) for radius calculation."
            )
        else:
            raise ValueError(f"Unsupported lens_shape: {self.lens_shape}")

        return float(r1), float(r2)

    def _create_surfaces(self, r1: float, r2: float):
        """
        Creates the two new standard Surface instances.
        Materials pre/post are set based on original paraxial surface context
        and the new lens material.
        """
        original_material_pre = self.paraxial_surface.material_pre
        original_material_post = self.paraxial_surface.material_post

        # Surface 1: front surface of the thick lens
        cs1 = CoordinateSystem()  # Will be updated by Optic during insertion
        geom1 = StandardGeometry(radius=r1, coordinate_system=cs1)
        surface1 = Surface(
            geometry=geom1,
            material_pre=original_material_pre,
            material_post=self._material_instance,
            is_stop=self.paraxial_surface.is_stop,
            comment="Thick Lens - Surface 1",
        )
        surface1.thickness = self.center_thickness  # Set thickness after instantiation

        # Surface 2: back surface of the thick lens
        cs2 = CoordinateSystem()  # Will be updated by Optic during insertion
        geom2 = StandardGeometry(radius=r2, coordinate_system=cs2)
        surface2 = Surface(
            geometry=geom2,
            material_pre=self._material_instance,
            material_post=original_material_post,
            is_stop=False,  # Stop, if any, is on the first surface
            comment="Thick Lens - Surface 2",
        )
        # surface2.thickness is initialized to 0.0 by Surface.__init__
        # and will be set correctly by _insert_new_surfaces

        # The Optic().add_surface method will handle clearing other stops if
        # surface1.is_stop is True when it's added.

        return surface1, surface2

    def _remove_paraxial_surface(self, original_index: int):
        """
        Removes the original ParaxialSurface from the parent optic's
        surface_group using its index.
        """
        # Basic check, SurfaceGroup.remove_surface has more robust checks
        if not (0 < original_index < len(self.optic.surface_group.surfaces)):
            raise IndexError(
                f"Invalid index {original_index} for removing paraxial surface."
            )
        self.optic.surface_group.remove_surface(original_index)

    def _insert_new_surfaces(
        self, surface1: Surface, surface2: Surface, original_index: int
    ):
        """
        Inserts the two new surfaces into the optic's surface_group.
        Adjusts positions for geometric center alignment.
        """
        original_paraxial_z_pos = self.paraxial_surface.geometry.cs.z

        # Thickness of surface2 (gap after it) is the original thickness
        # that was after the paraxial surface.
        surface2.thickness = self.paraxial_surface.thickness

        # Insert surfaces
        self.optic.add_surface(new_surface=surface1, index=original_index)
        self.optic.add_surface(new_surface=surface2, index=original_index + 1)

        # Adjust z-positions for center alignment.
        # This relies on optic.update() being called afterwards to correctly
        # propagate positions if relative positioning is used.
        # Desired z for surface1 vertex:
        desired_s1_z = original_paraxial_z_pos - (self.center_thickness / 2.0)

        if original_index > 0:
            # Adjust thickness of surface before s1
            surface_before_s1 = self.optic.surface_group.surfaces[original_index - 1]

            is_obj_at_inf = isinstance(
                surface_before_s1, self.optic.object_surface.__class__
            ) and be.isinf(surface_before_s1.thickness)

            if not is_obj_at_inf:
                # desired_s1_z = current_z_of_surface_before_s1 + new_thickness
                new_thickness_before_s1 = desired_s1_z - surface_before_s1.geometry.cs.z
                surface_before_s1.thickness = new_thickness_before_s1
            elif abs(desired_s1_z) > 1e-9 and is_obj_at_inf:
                # If object at inf, first optical surface is conventionally at z=0.
                # If desired_s1_z is non-zero, center alignment might be off
                # as optic.update() will likely place s1 at z=0.
                print(
                    f"Warning: Centering for thick lens after object@inf implies "
                    f"first surface at z={desired_s1_z}. Optic update may "
                    f"override this to z=0 if it's the first optical element."
                )
        else:  # surface1 is the new first optical element (original_index was 0)
            # This means paraxial_surface was at index 0, which is
            # unusual (object surface).
            # Assuming original_index refers to index *after* object surface.
            # If original_index is for the first *optical* surface (e.g. index 1):
            obj_surface = self.optic.surface_group.surfaces[0]  # Object surface
            if be.isinf(obj_surface.thickness):  # Object at infinity
                if abs(desired_s1_z) > 1e-9:
                    print(
                        f"Warning: Centering for thick lens as first element "
                        f"implies z={desired_s1_z}. Optic update may "
                        f"override this to z=0."
                    )
            else:  # Finite object distance
                # obj_surface.thickness is distance from object to first lens vertex
                obj_surface.thickness = desired_s1_z

        # Note: The final optic.update() in convert() is critical for these
        # thickness adjustments to correctly set all z-positions.
        pass


# Need to import Optic and Union for type hints if this class grows
# and gets moved to its own file. For now, keep it here and manage imports
# locally or defer full type hinting for Optic.
# from optiland.optic.optic import Optic # Causes circular import if not careful
