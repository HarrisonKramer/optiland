"""Grating Surface

This module defines the `GratingSurface` class, which represents a diffraction
grating in an optical system. This class extends the standard `Surface` to
include grating-specific parameters and behavior, such as diffraction order,
grating period, and groove orientation. It calculates the grating vector and
applies diffraction to rays interacting with the surface.

Kramer Harrison, 2023
"""

import optiland.backend as be
from optiland.geometries.base import BaseGeometry
from optiland.surfaces.standard_surface import Surface


class GratingSurface(Surface):
    """Represents a grating surface in an optical system.

    This class models a surface that diffracts light, defined by its geometry,
    material properties, and grating parameters.

    Args:
        geometry (BaseGeometry): The geometry of the surface.
        material_pre (BaseMaterial): The material before the surface.
        material_post (BaseMaterial): The material after the surface.
        grating_order (int): The diffraction order.
        grating_period (float): The distance between grating grooves.
        groove_orientation_angle (float, optional): The angle of the grating
            grooves in degrees. Defaults to 0.0.
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

    """

    def __init__(
        self,
        geometry: BaseGeometry,
        material_pre,
        material_post,
        grating_order: int,
        grating_period: float,
        groove_orientation_angle: float = 0.0,
        is_stop=False,
        aperture=None,
        coating=None,
        bsdf=None,
        is_reflective=False,
        surface_type="grating",
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
        )
        self.grating_order = be.array(grating_order)
        self.grating_period = be.array(grating_period)
        self.groove_orientation_angle = be.array(groove_orientation_angle)

    def _compute_tangent_to_grooves(self, rays, nx, ny, nz):
        """Computes a unit vector tangent to the grating grooves.

        This method calculates the direction of the grating grooves on the
        tangent plane of the surface. It establishes a local coordinate system
        on the tangent plane and rotates it by the groove orientation angle.

        Args:
            rays (RealRays): The rays interacting with the surface.
            nx (be.ndarray): The x-component of the surface normal vector.
            ny (be.ndarray): The y-component of the surface normal vector.
            nz (be.ndarray): The z-component of the surface normal vector.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The x, y, and z components
            of the unit vector tangent to the grooves.
        """
        # Choose a vector `s1` that is not parallel to the normal `n`
        s1_x = be.ones_like(nx)
        s1_y = be.zeros_like(ny)
        s1_z = be.zeros_like(nz)

        # Check for near-parallelism between n and s1
        dot_product = be.abs(nx * s1_x + ny * s1_y + nz * s1_z)
        is_parallel = dot_product > 0.999

        # If parallel, choose an alternative s1
        s1_x = be.where(is_parallel, be.zeros_like(s1_x), s1_x)
        s1_y = be.where(is_parallel, be.ones_like(s1_y), s1_y)

        # Create a tangent vector `t1` by taking the cross product of `n` and `s1`
        t1_x = ny * s1_z - nz * s1_y
        t1_y = nz * s1_x - nx * s1_z
        t1_z = nx * s1_y - ny * s1_x
        # Normalize t1
        norm_t1 = be.sqrt(t1_x**2 + t1_y**2 + t1_z**2)
        t1_x /= norm_t1
        t1_y /= norm_t1
        t1_z /= norm_t1

        # Create a second tangent vector `t2` orthogonal to `n` and `t1`
        t2_x = ny * t1_z - nz * t1_y
        t2_y = nz * t1_x - nx * t1_z
        t2_z = nx * t1_y - ny * t1_x

        # Rotate by the groove orientation angle
        phi = self.groove_orientation_angle
        cos_phi = be.cos(phi)
        sin_phi = be.sin(phi)
        t_groove_x = cos_phi * t1_x + sin_phi * t2_x
        t_groove_y = cos_phi * t1_y + sin_phi * t2_y
        t_groove_z = cos_phi * t1_z + sin_phi * t2_z

        return t_groove_x, t_groove_y, t_groove_z

    def _compute_grating_vector(self, rays, nx, ny, nz):
        """Computes the grating vector.

        The grating vector is perpendicular to both the surface normal and the
        grating grooves. It lies in the tangent plane of the surface.

        Args:
            rays (RealRays): The rays interacting with the surface.
            nx (be.ndarray): The x-component of the surface normal vector.
            ny (be.ndarray): The y-component of the surface normal vector.
            nz (be.ndarray): The z-component of the surface normal vector.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The x, y, and z components
            of the normalized grating vector.
        """
        # tangent to the grooves
        t_groove_x, t_groove_y, t_groove_z = self._compute_tangent_to_grooves(
            rays, nx, ny, nz
        )

        # grating vector is the cross product of normal and tangent
        fx = ny * t_groove_z - nz * t_groove_y
        fy = nz * t_groove_x - nx * t_groove_z
        fz = nx * t_groove_y - ny * t_groove_x

        # Normalize the grating vector
        norm_f = be.sqrt(fx**2 + fy**2 + fz**2)
        fx /= norm_f
        fy /= norm_f
        fz /= norm_f

        # Apply sign convention
        return -fx, -fy, -fz

    def _interact(self, rays):
        """Interacts the rays with the surface by diffracting them.

        Args:
            rays (RealRays): The rays to interact with the surface.

        Returns:
            RealRays: The diffracted rays.
        """
        # find surface normals
        nx, ny, nz = self.geometry.surface_normal(rays)

        # material refractive indices
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)

        # grating vector
        fx, fy, fz = self._compute_grating_vector(rays, nx, ny, nz)

        # diffraction order and period
        m = self.grating_order
        d = self.grating_period

        # apply diffraction
        rays.gratingdiffract(nx, ny, nz, fx, fy, fz, m, d, n1, n2, self.is_reflective)

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

    def to_dict(self):
        """Convert the grating surface to a dictionary."""
        d = super().to_dict()
        d.update(
            {
                "grating_order": self.grating_order,
                "grating_period": self.grating_period,
                "groove_orientation_angle": self.groove_orientation_angle,
            }
        )
        return d

    @classmethod
    def _from_dict(cls, data):
        """Create a GratingSurface instance from a dictionary."""
        surface = super()._from_dict(data)
        surface.grating_order = data.get("grating_order")
        surface.grating_period = data.get("grating_period")
        surface.groove_orientation_angle = data.get("groove_orientation_angle", 0.0)
        return surface
