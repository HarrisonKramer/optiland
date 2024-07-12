"""Optiland Surfaces Module

This module defines the `Surface` class, which represents a surface in an
optical system. Surfaces are characterized by their geometry, materials before
and after the surface, and optional properties such as being an aperture stop,
having a physical aperture, and a coating. The module facilitates the tracing
of rays through these surfaces, accounting for refraction, reflection, and
absorption based on the surface properties and materials involved.

Kramer Harrison, 2023
"""
from typing import List
from copy import deepcopy
import numpy as np
from optiland.rays import BaseRays, RealRays, ParaxialRays
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import Plane, StandardGeometry, BaseGeometry
from optiland.materials import BaseMaterial, IdealMaterial, Material
from optiland.physical_apertures import BaseAperture
from optiland.coatings import BaseCoating


class Surface:
    """
    Represents a standard refractice surface in an optical system.

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
    """

    def __init__(self,
                 geometry: BaseGeometry,
                 material_pre: BaseMaterial,
                 material_post: BaseMaterial,
                 is_stop: bool = False,
                 aperture: BaseAperture = None,
                 coating: BaseCoating = None):
        self.geometry = geometry
        self.material_pre = material_pre
        self.material_post = material_post
        self.is_stop = is_stop
        self.aperture = aperture
        self.semi_aperture = None
        self.coating = coating

        self._is_reflective = False

        self.reset()

    def trace(self, rays: BaseRays):
        """
        Traces the given rays through the surface.

        Args:
            rays (BaseRays): The rays to be traced.

        Returns:
            BaseRays: The traced rays.
        """
        if isinstance(rays, ParaxialRays):
            return self._trace_paraxial(rays)
        elif isinstance(rays, RealRays):
            return self._trace_real(rays)

    def set_semi_aperture(self, r_max: float):
        """
        Sets the physical semi-aperture of the surface.

        Args:
            r_max (float): The maximum radius of the semi-aperture.
        """
        self.semi_aperture = r_max

    def reset(self):
        """
        Resets the recorded information of the surface.
        """
        self.y = np.empty(0)
        self.u = np.empty(0)
        self.x = np.empty(0)
        self.y = np.empty(0)
        self.z = np.empty(0)

        self.L = np.empty(0)
        self.M = np.empty(0)
        self.N = np.empty(0)

        self.energy = np.empty(0)
        self.aoi = np.empty(0)
        self.opd = np.empty(0)

    def _compute_aoi(self, rays, nx, ny, nz):
        """
        Computes the angle of incidence for the given rays and surface normals.

        Args:
            rays: The rays.
            nx: The x-component of the surface normals.
            ny: The y-component of the surface normals.
            nz: The z-component of the surface normals.

        Returns:
            np.ndarray: The angle of incidence for each ray.
        """
        dot = np.abs(nx * rays.L + ny * rays.M + nz * rays.N)
        dot = np.clip(dot, -1, 1)  # required due to numerical precision
        return np.arccos(dot)

    def _record(self, rays):
        """
        Records the ray information.

        Args:
            rays: The rays.
        """
        if isinstance(rays, ParaxialRays):
            self.y = np.copy(np.atleast_1d(rays.y))
            self.u = np.copy(np.atleast_1d(rays.u))
        elif isinstance(rays, RealRays):
            self.x = np.copy(np.atleast_1d(rays.x))
            self.y = np.copy(np.atleast_1d(rays.y))
            self.z = np.copy(np.atleast_1d(rays.z))

            self.L = np.copy(np.atleast_1d(rays.L))
            self.M = np.copy(np.atleast_1d(rays.M))
            self.N = np.copy(np.atleast_1d(rays.N))

            self.energy = np.copy(np.atleast_1d(rays.e))
            self.opd = np.copy(np.atleast_1d(rays.opd))

    def _interact(self, rays):
        """
        Interacts the rays with the surface by refracting them.

        Args:
            rays: The rays.

        Returns:
            RealRays: The refracted rays.
        """
        # find surface normals
        nx, ny, nz = self.geometry.surface_normal(rays)

        # Get initial ray parameters for coating
        if self.coating:
            params = self.coating.create_interaction_params(rays)

        # Interact with surface (refract or reflect)
        if self._is_reflective:
            rays = self._reflect(rays, nx, ny, nz)
        else:
            rays = self._refract(rays, nx, ny, nz)

        # if there is a coating, modify ray properties
        if self.coating:
            params.rays = rays  # assign rays after refraction
            rays = self.coating.interact(params, reflect=False)

        return rays

    def _refract(self, rays, nx, ny, nz):
        """
        Refract rays on the surface.

        Args:
            rays: The rays.
            nx: The x-component of the surface normals.
            ny: The y-component of the surface normals.
            nz: The z-component of the surface normals.

        Returns:
            RealRays: The refracted rays.
        """
        ix = rays.L
        iy = rays.M
        iz = rays.N

        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)

        u = n1 / n2
        ni = nx*ix + ny*iy + nz*iz
        root = np.sqrt(1 - u**2 * (1 - ni**2))
        tx = u * ix + nx * root - u * nx * ni
        ty = u * iy + ny * root - u * ny * ni
        tz = u * iz + nz * root - u * nz * ni

        rays.L = tx
        rays.M = ty
        rays.N = tz

        return rays

    def _reflect(self, rays, nx, ny, nz):
        """
        Reflects the rays on the surface.

        Args:
            rays: The rays to be reflected.
            nx: The x-component of the surface normal.
            ny: The y-component of the surface normal.
            nz: The z-component of the surface normal.

        Returns:
            RealRays: The reflected rays.
        """
        dot = rays.L * nx + rays.M * ny + rays.N * nz
        rays.L -= 2 * dot * nx
        rays.M -= 2 * dot * ny
        rays.N -= 2 * dot * nz
        return rays

    def _trace_paraxial(self, rays: ParaxialRays):
        """
        Traces paraxial rays through the surface.

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

        # surface power
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)
        power = (n2 - n1) / self.geometry.radius

        # refract
        rays.u = 1 / n2 * (n1 * rays.u - rays.y * power)

        # inverse transform coordinate system
        self.geometry.globalize(rays)

        self._record(rays)

    def _trace_real(self, rays: RealRays):
        """
        Traces real rays through the surface.

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

        # propagate the rays a distance t
        rays.propagate(t)

        # update OPD
        rays.opd += np.abs(t * self.material_pre.n(rays.w))

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


class ReflectiveSurface(Surface):
    """
    A class representing a reflective surface.

    Inherits from the Surface class and provides methods for reflecting rays
    on the surface.

    Args:
        geometry (BaseGeometry): The geometry of the surface.
        material_pre (BaseMaterial): The material before the surface.
        is_stop (bool, optional): Indicates if the surface is the aperture
            stop. Defaults to False.
        aperture (float, optional): The physical aperture of the surface.
            Defaults to None.
    """

    def __init__(self, geometry: BaseGeometry, material_pre: BaseMaterial,
                 is_stop: bool = False, aperture: BaseAperture = None):
        super().__init__(
            geometry=geometry,
            material_pre=material_pre,
            material_post=material_pre,
            is_stop=is_stop,
            aperture=aperture
        )

        self._is_reflective = True

    def _trace_paraxial(self, rays: ParaxialRays):
        """
        Trace paraxial rays through the surface.

        Args:
            rays (ParaxialRays): The paraxial rays to be traced.

        Returns:
            None

        This method traces the given paraxial rays through the surface.
        It performs the following steps:
            1. Resets the recorded information.
            2. Localizes the coordinate system based on the surface geometry.
            3. Propagates the rays to the surface.
            4. Reflects the rays using the paraxial equations.
            5. Globalizes the coordinate system based on the surface geometry.
            6. Records the traced rays.

        Note:
            The paraxial rays are modified in-place.
            The surface geometry must be set before calling this method.
        """
        # reset recorded information
        self.reset()

        # transform coordinate system
        self.geometry.localize(rays)

        # propagate to this surface
        t = -rays.z
        rays.propagate(t)

        # reflect (derived from paraxial equations when n'=-n)
        rays.u = -rays.u - 2 * rays.y / self.geometry.radius

        # inverse transform coordinate system
        self.geometry.globalize(rays)

        self._record(rays)


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

    def _interact(self, rays, nx, ny, nz):
        """
        Interacts the given rays with the surface.

        Args:
            rays (Rays): The rays to be interacted.
            nx (float): The x-component of the surface normal.
            ny (float): The y-component of the surface normal.
            nz (float): The z-component of the surface normal.

        Returns:
            RealRays: The interacted rays.
        """
        return rays


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
                 aperture: BaseAperture = None):
        super().__init__(
            geometry=geometry,
            material_pre=material_pre,
            material_post=material_pre,
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

        self._record(rays)

    def _interact(self, rays, nx, ny, nz):
        """
        Interacts rays with the surface.

        Args:
            rays: The rays to be interacted with the surface.
            nx: The x-component of the surface normal.
            ny: The y-component of the surface normal.
            nz: The z-component of the surface normal.

        Returns:
            RealRays: The modified rays after interaction with the surface.
        """
        return rays


class SurfaceGroup:
    """
    Represents a group of surfaces in an optical system.

    Attributes:
        surfaces (list): List of surfaces in the group.
        _last_thickness (float): The thickness of the last surface added.
    """

    def __init__(self, surfaces: List[Surface] = None):
        """
        Initializes a new instance of the SurfaceGroup class.

        Args:
            surfaces (List, optional): List of surfaces to initialize the
                group with. Defaults to None.
        """
        if surfaces is None:
            self.surfaces = []
        else:
            self.surfaces = surfaces

        self._last_thickness = 0

    @property
    def x(self):
        """np.array: x intersection points on all surfaces"""
        return np.array([surf.x for surf in self.surfaces if surf.x.size > 0])

    @property
    def y(self):
        """np.array: y intersection points on all surfaces"""
        return np.array([surf.y for surf in self.surfaces if surf.y.size > 0])

    @property
    def z(self):
        """np.array: z intersection points on all surfaces"""
        return np.array([surf.z for surf in self.surfaces if surf.z.size > 0])

    @property
    def L(self):
        """np.array: x direction cosines on all surfaces"""
        return np.array([surf.L for surf in self.surfaces if surf.L.size > 0])

    @property
    def M(self):
        """np.array: y direction cosines on all surfaces"""
        return np.array([surf.M for surf in self.surfaces if surf.M.size > 0])

    @property
    def N(self):
        """np.array: z direction cosines on all surfaces"""
        return np.array([surf.N for surf in self.surfaces if surf.N.size > 0])

    @property
    def opd(self):
        """np.array: optical path difference recorded on all surfaces"""
        return np.array([surf.opd for surf in self.surfaces
                         if surf.opd.size > 0])

    @property
    def u(self):
        """np.array: paraxial ray angles on all surfaces"""
        return np.array([surf.u for surf in self.surfaces if surf.u.size > 0])

    @property
    def energy(self):
        """np.array: ray energies on all surfaces"""
        return np.array([surf.energy for surf in self.surfaces
                         if surf.energy.size > 0])

    @property
    def positions(self):
        """np.array: z positions of surface vertices"""
        return np.array([surf.geometry.cs.position_in_gcs[2]
                         for surf in self.surfaces])

    @property
    def radii(self):
        """np.array: radii of curvature of all surfaces"""
        return np.array([surf.geometry.radius for surf in self.surfaces])

    @property
    def conic(self):
        """np.array: conic constant of all surfaces"""
        values = []
        for surf in self.surfaces:
            try:
                values.append(surf.geometry.k)
            except AttributeError:
                values.append(0)
        return np.array(values)

    @property
    def stop_index(self):
        """int: the index of the aperture stop surface"""
        for index, surface in enumerate(self.surfaces):
            if surface.is_stop:
                return index

    @property
    def num_surfaces(self):
        """int: the number of surfaces"""
        return len(self.surfaces)

    def get_thickness(self, surface_number):
        """
        Calculate the thickness between two surfaces.

        Args:
            surface_number (int): The index of the first surface.

        Returns:
            float: The thickness between the two surfaces.
        """
        t = self.positions
        return t[surface_number+1] - t[surface_number]

    def trace(self, rays, skip=0):
        """
        Trace the given rays through the surfaces.

        Args:
            rays (BaseRays): List of rays to be traced.
            skip (int, optional): Number of surfaces to skip before tracing.
                Defaults to 0.
        """
        self.reset()
        for surface in self.surfaces[skip:]:
            surface.trace(rays)

    def add_surface(self, new_surface=None, index=None, thickness=0,
                    radius=np.inf, material='air', conic=0, is_stop=False,
                    dx=0, dy=0, rx=0, ry=0, aperture=None):
        """
        Adds a new surface to the list of surfaces.

        Args:
            new_surface (Surface, optional): The new surface to add. If not
                provided, a new surface will be created based on the other
                arguments.
            index (int, optional): The index at which to insert the new
                surface. If not provided, the surface will be appended to the
                end of the list.
            thickness (float, optional): The thickness of the surface.
                Default is 0.
            radius (float, optional): The radius of curvature of the surface.
                Default is infinity.
            material (str, optional): The material of the surface.
                Default is 'air'.
            conic (float, optional): The conic constant of the surface.
                Default is 0.
            is_stop (bool, optional): Whether the surface is the aperture stop
                surface. Default is False.
            dx (float, optional): The x-coordinate displacement of the
                surface. Default is 0.
            dy (float, optional): The y-coordinate displacement of the
                surface. Default is 0.
            rx (float, optional): The x-axis rotation angle of the surface.
                Default is 0.
            ry (float, optional): The y-axis rotation angle of the surface.
                Default is 0.
            aperture (BaseAperture, optional): The physical aperture of the
                surface. Default is None.

        Raises:
            ValueError: If index is not provided when defining a new surface.

        """
        if new_surface is None:
            if index is None:
                raise ValueError('Must define index when defining surface.')

            new_surface = self._configure_surface(index, thickness, radius,
                                                  material, conic, is_stop,
                                                  dx, dy, rx, ry, aperture)

        if new_surface.is_stop:
            for surface in self.surfaces:
                surface.is_stop = False

        self.surfaces.insert(index, new_surface)

    def remove_surface(self, index):
        """
        Remove a surface from the list of surfaces.

        Args:
            index (int): The index of the surface to remove.

        Raises:
            ValueError: If the index is 0 (object surface).

        Returns:
        None
        """
        if index == 0:
            raise ValueError('Cannot remove object surface.')
        del self.surfaces[index]

    def reset(self):
        """
        Resets all the surfaces in the collection.

        This method iterates over each surface in the collection and calls
            its `reset` method.
        """
        for surface in self.surfaces:
            surface.reset()

    def inverted(self):
        """Generate inverted surface group.

        This method generates an inverted surface group by performing the
            following operations:
            1. Reverses the order of the surfaces in the original surface
                group.
            2. Scales the radii of each surface by -1.
            3. Inverts the z position of each surface by subtracting it from
                the z position of the last surface.
            4. Swaps the initial and final materials of each surface.

        Returns:
            SurfaceGroup: The inverted surface group.

        """
        surfs_inverted = deepcopy(self.surfaces[::-1])
        z_shift = self.surfaces[-1].geometry.cs.z
        for surf in surfs_inverted:
            # scale radii by -1
            surf.geometry.radius *= -1

            # invert z position
            surf.geometry.cs.z = z_shift - surf.geometry.cs.z

            # swap initial and final materials
            temp = surf.material_pre
            surf.material_pre = surf.material_post
            surf.material_post = temp

        return SurfaceGroup(surfs_inverted)

    def _configure_cs(self, index, thickness, dx, dy, rx, ry):
        """
        Configures the coordinate system for a given surface.

        Args:
            index (int): The index of the surface.
            thickness (float): The thickness of the surface.
            dx (float): The x-coordinate offset of the surface.
            dy (float): The y-coordinate offset of the surface.
            rx (float): The rotation around the x-axis of the surface.
            ry (float): The rotation around the y-axis of the surface.

        Returns:
            CoordinateSystem: The configured coordinate system.
        """
        if index == 0:  # object surface
            z = -thickness
        elif index == 1:
            z = 0  # first surface, always at zero
        else:
            z = self.positions[index-1] + self._last_thickness

        self._last_thickness = thickness

        return CoordinateSystem(x=dx, y=dy, z=z, rx=rx, ry=ry)

    def _configure_geometry(self, cs, radius, conic):
        """
        Configures the geometry based on the given parameters.

        Parameters:
            cs: The coordinate system for the geometry.
            radius: The radius of the geometry. If it is infinity, a plane
                geometry is used.
            conic: The conic constant for the geometry.

        Returns:
            geometry: The configured geometry object.

        """
        if np.isinf(radius):
            geometry = Plane(cs)
        else:
            geometry = StandardGeometry(cs, radius, conic)

        return geometry

    def _configure_material(self, index, material):
        """
        Configures the material for a surface based on the given index and
            material input.

        Args:
            index (int): The index of the surface.
            material (BaseMaterial, tuple, str): The material input for the
                surface. It can be an instance of BaseMaterial, a tuple
                containing the name and reference of the material, or a string
                representing the material. See examples.

        Returns:
            tuple: A tuple containing the material before and after the
                surface.
        """
        if isinstance(material, BaseMaterial):
            material_post = material
        elif isinstance(material, tuple):
            material_post = Material(name=material[0], reference=material[1])
        elif isinstance(material, str):
            if material in ['mirror', 'air']:
                material_post = IdealMaterial(n=1.0, k=0.0)
            else:
                material_post = Material(material)

        if index == 0:
            material_pre = None
        else:
            material_pre = self.surfaces[index-1].material_post

        return material_pre, material_post

    def _configure_surface(self, index, thickness=0, radius=np.inf,
                           material='air', conic=0, is_stop=False,
                           dx=0, dy=0, rx=0, ry=0, aperture=None):
        """
        Configures a surface based on the provided parameters.

        Args:
            index (int): The index of the surface.
            thickness (float, optional): The thickness of the surface.
                Defaults to 0.
            radius (float, optional): The radius of curvature of the surface.
                Defaults to np.inf.
            material (str, optional): The material of the surface.
                Defaults to 'air'.
            conic (float, optional): The conic constant of the surface.
                Defaults to 0.
            is_stop (bool, optional): Indicates if the surface is the aperture
                stop. Defaults to False.
            dx (float, optional): The x-coordinate displacement of the
                surface. Defaults to 0.
            dy (float, optional): The y-coordinate displacement of the
                surface. Defaults to 0.
            rx (float, optional): The x-axis rotation angle of the surface.
                Defaults to 0.
            ry (float, optional): The y-axis rotation angle of the surface.
                Defaults to 0.
            aperture (float, optional): The physical aperture of the surface.
                Defaults to None.

        Returns:
            Surface: The configured surface object.

        Raises:
            ValueError: If the surface index is greater than the number of
                surfaces.
        """
        if index > self.num_surfaces:
            raise ValueError('Surface index cannot be greater than number of '
                             'surfaces.')

        cs = self._configure_cs(index, thickness, dx, dy, rx, ry)
        geometry = self._configure_geometry(cs, radius, conic)
        material_pre, material_post = self._configure_material(index, material)

        if index == 0:
            return ObjectSurface(geometry, material_post)
        elif index == self.num_surfaces-1:
            return ImageSurface(geometry, material_pre, aperture)
        else:
            if material == 'mirror':
                return ReflectiveSurface(geometry, material_pre, is_stop,
                                         aperture)
            else:
                return Surface(geometry, material_pre, material_post, is_stop,
                               aperture)
