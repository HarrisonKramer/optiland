"""Grating Surface

This module defines the `GratingSurface` class, which represents a diffraction
grating in an optical system. This class extends the standard `Surface` and
encapsulates all logic related to grating diffraction, including the definition
of grating parameters (order and period) and the computation of the grating vector.

Matteo Taccola & Kramer Harrison, 2025
"""

import optiland.backend as be
from optiland.geometries.base import BaseGeometry
from optiland.rays import RealRays
from optiland.surfaces.standard_surface import Surface


class GratingSurface(Surface):
    """Represents a grating surface in an optical system.

    This class models a surface that diffracts light, defined by its geometry,
    material properties, and grating parameters. It is responsible for
    calculating the grating vector and applying diffraction to rays interacting
    with the surface, independent of the specific geometry's internal grating
    definitions.

    Args:
        geometry (BaseGeometry): The geometry of the surface. This object
            is used solely to obtain the surface normal.
        material_pre (BaseMaterial): The material before the surface.
        material_post (BaseMaterial): The material after the surface.
        grating_order (int): The diffraction order for the grating.
        grating_period (float): The distance between grating grooves.
        is_stop (bool, optional): Indicates if the surface is the aperture
            stop. Defaults to False.
        aperture (BaseAperture, int, float, optional): The physical aperture of
            the surface. Defaults to None.
        coating (BaseCoating, optional): The coating applied to the surface.
            Defaults to None.
        bsdf (BaseBSDF, optional): The bidirectional scattering distribution
            function. Defaults to None.
        is_reflective (bool, optional): Indicates if the surface is reflective.
            Defaults to False.
        surface_type (str, optional): The type of the surface. Defaults to "grating".
        comment (str, optional): A comment for the surface. Defaults to ''.

    Attributes:
        grating_order (be.ndarray): The diffraction order.
        grating_period (be.ndarray): The distance between grating grooves.
    """

    def __init__(
        self,
        geometry: BaseGeometry,
        material_pre,
        material_post,
        grating_order: int,
        grating_period: float,
        is_stop: bool = False,
        aperture=None,
        coating=None,
        bsdf=None,
        is_reflective: bool = False,
        surface_type: str = "grating",
        comment: str = "",
    ):
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
        self.grating_order = be.array(grating_order)
        self.grating_period = be.array(grating_period)

    def _compute_tangent_to_grooves(self, nx, ny, nz):
        """Computes a unit vector tangent to the grating grooves.

        This method calculates the direction of the grating grooves on the
        tangent plane of the surface. It establishes a local coordinate system
        on the tangent plane and assumes a fixed orientation for the grooves
        relative to this system, consistent with the original implementation's
        implicit behavior (without a `groove_orientation_angle` parameter).

        Args:
            nx (be.ndarray): The x-component of the surface normal vector.
            ny (be.ndarray): The y-component of the surface normal vector.
            nz (be.ndarray): The z-component of the surface normal vector.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The x, y, and z components
            of the unit vector tangent to the grooves.
        """
        # Choose a vector `s1` that is not parallel to the normal `n`.
        # Default to (1, 0, 0). If n is parallel to (1,0,0), use (0,1,0).
        s1_x = be.ones_like(nx)
        s1_y = be.zeros_like(ny)
        s1_z = be.zeros_like(nz)

        # Check for near-parallelism between n and s1
        dot_product = be.abs(nx * s1_x + ny * s1_y + nz * s1_z)
        is_parallel = dot_product > 0.999

        # If parallel, choose an alternative s1 (e.g., (0, 1, 0))
        s1_x = be.where(is_parallel, be.zeros_like(s1_x), s1_x)
        s1_y = be.where(is_parallel, be.ones_like(s1_y), s1_y)
        s1_z = be.where(is_parallel, be.zeros_like(s1_z), s1_z)

        # Create a tangent vector `t1` by taking the cross product of `n` and `s1`
        t1_x = ny * s1_z - nz * s1_y
        t1_y = nz * s1_x - nx * s1_z
        t1_z = nx * s1_y - ny * s1_x

        # Normalize t1
        norm_t1 = be.sqrt(t1_x**2 + t1_y**2 + t1_z**2)

        # Handle cases where norm_t1 might be zero
        t1_x = be.where(norm_t1 != 0, t1_x / norm_t1, t1_x)
        t1_y = be.where(norm_t1 != 0, t1_y / norm_t1, t1_y)
        t1_z = be.where(norm_t1 != 0, t1_z / norm_t1, t1_z)

        # Create a second tangent vector `t2` orthogonal to `n` and `t1`
        t2_x = ny * t1_z - nz * t1_y
        t2_y = nz * t1_x - nx * t1_z
        t2_z = nx * t1_y - ny * t1_x

        return t2_x, t2_y, t2_z

    def _compute_grating_vector(self, nx, ny, nz):
        """Computes the grating vector based on the surface normal.

        The grating vector is perpendicular to both the surface normal and the
        grating grooves. It lies in the tangent plane of the surface. This
        method directly computes the grating vector using the surface normal
        and an implicitly defined groove orientation.

        Args:
            nx (be.ndarray): The x-component of the surface normal vector.
            ny (be.ndarray): The y-component of the surface normal vector.
            nz (be.ndarray): The z-component of the surface normal vector.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The x, y, and z components
            of the normalized grating vector.
        """
        # Get the tangent vector to the grooves based on the surface normal.
        t_groove_x, t_groove_y, t_groove_z = self._compute_tangent_to_grooves(
            nx, ny, nz
        )

        # Grating vector is cross product of the normal and the tangent to the grooves.
        fx = ny * t_groove_z - nz * t_groove_y
        fy = nz * t_groove_x - nx * t_groove_z
        fz = nx * t_groove_y - ny * t_groove_x

        # Normalize the grating vector
        norm_f = be.sqrt(fx**2 + fy**2 + fz**2)

        # Avoid division by zero
        fx = be.where(norm_f != 0, fx / norm_f, fx)
        fy = be.where(norm_f != 0, fy / norm_f, fy)
        fz = be.where(norm_f != 0, fz / norm_f, fz)

        return fx, fy, fz

    def _interact(self, rays: RealRays) -> RealRays:
        """Interacts the rays with the surface by diffracting them.

        This method is the core of the grating interaction. It retrieves the
        surface normal from the geometry, computes the grating vector internally,
        applies a projection correction to the grating period, and then
        calls the ray diffraction routine.

        Args:
            rays (RealRays): The rays to interact with the surface.

        Returns:
            RealRays: The diffracted rays.
        """
        # Find surface normals from the geometry.
        nx, ny, nz = self.geometry.surface_normal(rays)

        # Get material refractive indices.
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)

        # Compute the grating vector.
        fx, fy, fz = self._compute_grating_vector(nx, ny, nz)

        # Get diffraction order and period.
        m = self.grating_order
        d = self.grating_period

        # Apply the grating period correction. This accounts for the
        # projection of the grating period onto the plane perpendicular to the
        # grating vector's x-y components.
        norm_f_xy = be.sqrt(fx**2 + fy**2)

        # Avoid division by zero if the x-y components of the grating vector are zero.
        d_corrected = be.where(norm_f_xy != 0, d / norm_f_xy, d)

        # Apply diffraction using the corrected period and computed grating vector.
        rays.gratingdiffract(
            nx, ny, nz, fx, fy, fz, m, d_corrected, n1, n2, self.is_reflective
        )

        # If there is a surface scatter model, modify ray properties.
        if self.bsdf:
            rays = self.bsdf.scatter(rays, nx, ny, nz)

        # If there is a coating, modify ray properties.
        if self.coating:
            rays = self.coating.interact(
                rays,
                reflect=self.is_reflective,
                nx=nx,
                ny=ny,
                nz=nz,
            )
        else:
            # Update polarization matrices, if PolarizedRays.
            rays.update()

        return rays

    def to_dict(self) -> dict:
        """Convert the grating surface to a dictionary.

        Returns:
            dict: The dictionary representation of the grating surface,
                including its grating order and period.
        """
        d = super().to_dict()
        d.update(
            {
                "grating_order": float(self.grating_order),
                "grating_period": float(self.grating_period),
            }
        )
        return d

    @classmethod
    def _from_dict(cls, data: dict) -> "GratingSurface":
        """Protected deserialization logic for direct initialization.

        Creates a GratingSurface instance from a dictionary.

        Args:
            data (dict): The dictionary representation of the surface.

        Returns:
            GratingSurface: The grating surface.

        Raises:
            ValueError: If 'grating_order' or 'grating_period' are missing
                from the dictionary data.
        """
        base_surface = super()._from_dict(data)

        grating_order = data.get("grating_order")
        grating_period = data.get("grating_period")

        return cls(
            geometry=base_surface.geometry,
            material_pre=base_surface.material_pre,
            material_post=base_surface.material_post,
            grating_order=grating_order,
            grating_period=grating_period,
            is_stop=base_surface.is_stop,
            aperture=base_surface.aperture,
            coating=base_surface.coating,
            bsdf=base_surface.bsdf,
            is_reflective=base_surface.is_reflective,
            surface_type=base_surface.surface_type,
            comment=base_surface.comment,
        )
