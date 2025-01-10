"""Optic Module

This is the core module of Optiland, which provides the class to define
generic optical systems. The Optic class encapsulates the core properties
of an optical system, such as the aperture, fields, surfaces, and
wavelengths. It also provides methods to draw the optical system, trace rays,
and perform paraxial and aberration analyses. Instances of the Optic class
are used as arguments to various analysis, optimization, and visualization
functions in Optiland.

Kramer Harrison, 2024
"""
from typing import Union
import numpy as np
from optiland.fields import Field, FieldGroup
from optiland.surfaces import SurfaceGroup, ObjectSurface
from optiland.wavelength import WavelengthGroup
from optiland.paraxial import Paraxial
from optiland.aberrations import Aberrations
from optiland.aperture import Aperture
from optiland.rays import PolarizedRays, PolarizationState, RayGenerator
from optiland.distribution import create_distribution
from optiland.geometries import Plane, StandardGeometry
from optiland.materials import IdealMaterial
from optiland.visualization import OpticViewer, OpticViewer3D, LensInfoViewer
from optiland.pickup import PickupManager
from optiland.solves import SolveManager


class Optic:
    """
    The Optic class represents an optical system.

    Attributes:
        aperture (Aperture): The aperture of the optical system.
        field_type (str): The type of field used in the optical system.
        surface_group (SurfaceGroup): The group of surfaces in the optical
            system.
        fields (FieldGroup): The group of fields in the optical system.
        wavelengths (WavelengthGroup): The group of wavelengths in the optical
            system.
        paraxial (Paraxial): The paraxial analysis helper class for the
            optical system.
        aberrations (Aberrations): The aberrations analysis helper class for
            the optical system.
    """

    def __init__(self):
        self.aperture = None
        self.field_type = None

        self.surface_group = SurfaceGroup()
        self.fields = FieldGroup()
        self.wavelengths = WavelengthGroup()

        self.paraxial = Paraxial(self)
        self.aberrations = Aberrations(self)
        self.ray_generator = RayGenerator(self)

        self.polarization = 'ignore'

        self.pickups = PickupManager(self)
        self.solves = SolveManager(self)
        self.obj_space_telecentric = False

    @property
    def primary_wavelength(self):
        """float: the primary wavelength in microns"""
        return self.wavelengths.primary_wavelength.value

    @property
    def object_surface(self):
        """Surface: the object surface instance"""
        for surface in self.surface_group.surfaces:
            if isinstance(surface, ObjectSurface):
                return surface
        return None

    @property
    def image_surface(self):
        """Surface: the image surface instance"""
        return self.surface_group.surfaces[-1]

    @property
    def total_track(self):
        """float: the total track length of the system"""
        z = self.surface_group.positions[1:-1]
        return np.max(z) - np.min(z)

    @property
    def polarization_state(self):
        """PolarizationState: the polarization state of the optic"""
        if self.polarization == 'ignore':
            return None
        elif isinstance(self.polarization, PolarizationState):
            return self.polarization

    def add_surface(self, new_surface=None, surface_type='standard',
                    index=None, is_stop=False, material='air', **kwargs):
        """
        Adds a new surface to the optic.

        Args:
            new_surface (Surface, optional): The new surface to add. If not
                provided, a new surface will be created based on the other
                arguments.
            surface_type (str, optional): The type of surface to create.
            index (int, optional): The index at which to insert the new
                surface. If not provided, the surface will be appended to the
                end of the list.
            is_stop (bool, optional): Indicates if the surface is the aperture.
            material (str, optional): The material of the surface.
                Default is 'air'.
            **kwargs: Additional keyword arguments for surface-specific
                parameters such as radius, conic, dx, dy, rx, ry, aperture.

        Raises:
            ValueError: If index is not provided when defining a new surface.
        """
        self.surface_group.add_surface(
            new_surface=new_surface, surface_type=surface_type, index=index,
            is_stop=is_stop, material=material, **kwargs
            )

    def add_field(self, y, x=0.0, vx=0.0, vy=0.0):
        """
        Add a field to the optical system.

        Args:
            y (float): The y-coordinate of the field.
            x (float, optional): The x-coordinate of the field.
                Defaults to 0.0.
            vx (float, optional): The x-component of the field's vignetting
                factor. Defaults to 0.0.
            vy (float, optional): The y-component of the field's vignetting
                factor. Defaults to 0.0.
        """
        new_field = Field(self.field_type, x, y, vx, vy)
        self.fields.add_field(new_field)

    def add_wavelength(self, value, is_primary=False, unit='um'):
        """
        Add a wavelength to the optical system.

        Args:
            value (float): The value of the wavelength.
            is_primary (bool, optional): Whether the wavelength is the primary
                wavelength. Defaults to False.
            unit (str, optional): The unit of the wavelength. Defaults to 'um'.
        """
        self.wavelengths.add_wavelength(value, is_primary, unit)

    def set_aperture(self, aperture_type, value):
        """
        Set the aperture of the optical system.

        Args:
            aperture_type (str): The type of the aperture.
            value (float): The value of the aperture.
        """
        self.aperture = Aperture(aperture_type, value)

    def set_field_type(self, field_type):
        """
        Set the type of field used in the optical system.

        Args:
            field_type (str): The type of field.
        """
        if field_type not in ['angle', 'object_height']:
            raise ValueError('Invalid field type. Must be "angle" '
                             'or "object_height".')
        self.field_type = field_type

    def set_radius(self, value, surface_number):
        """
        Set the radius of curvature of a surface.

        Args:
            value (float): The value of the radius.
            surface_number (int): The index of the surface.
        """
        surface = self.surface_group.surfaces[surface_number]

        # change geometry from plane to standard
        if isinstance(surface.geometry, Plane):
            cs = surface.geometry.cs
            new_geometry = StandardGeometry(cs, radius=value, conic=0)
            surface.geometry = new_geometry
        else:
            surface.geometry.radius = value

    def set_conic(self, value, surface_number):
        """
        Set the conic constant of a surface.

        Args:
            value (float): The value of the conic constant.
            surface_number (int): The index of the surface.
        """
        surface = self.surface_group.surfaces[surface_number]
        surface.geometry.k = value

    def set_thickness(self, value, surface_number):
        """
        Set the thickness of a surface.

        Args:
            value (float): The value of the thickness.
            surface_number (int): The index of the surface.
        """
        positions = self.surface_group.positions
        delta_t = value - positions[surface_number+1] + \
            positions[surface_number]
        positions[surface_number+1:] += delta_t
        positions -= positions[1]  # force surface 1 to be at zero
        for k, surface in enumerate(self.surface_group.surfaces):
            surface.geometry.cs.z = positions[k]

    def set_index(self, value, surface_number):
        """
        Set the index of refraction of a surface.

        Args:
            value (float): The value of the index of refraction.
            surface_number (int): The index of the surface.
        """
        surface = self.surface_group.surfaces[surface_number]
        new_material = IdealMaterial(n=value, k=0)
        surface.material_post = new_material

        surface_post = self.surface_group.surfaces[surface_number+1]
        surface_post.material_pre = new_material

    def set_asphere_coeff(self, value, surface_number, aspher_coeff_idx):
        """
        Set the asphere coefficient on a surface

        Args:
            value (float): The value of aspheric coefficient
            surface_number (int): The index of the surface.
            aspher_coeff_idx (int): index of the aspheric coefficient on the
                surface
        """
        surface = self.surface_group.surfaces[surface_number]
        surface.geometry.c[aspher_coeff_idx] = value

    def set_polarization(self, polarization: Union[PolarizationState, str]):
        """
        Set the polarization state of the optic.

        Parameters:
            polarization (Union[PolarizationState, str]): The polarization
                state to set. It can be either a `PolarizationState` object or
                'ignore'.
        """
        if isinstance(polarization, str) and polarization != 'ignore':
            raise ValueError('Invalid polarization state. Must be either '
                             'PolarizationState or "ignore".')
        self.polarization = polarization

    def scale_system(self, scale_factor):
        """
        Scales the optical system by a given scale factor.

        Args:
            scale_factor (float): The factor by which to scale the system.
        """
        num_surfaces = self.surface_group.num_surfaces
        radii = self.surface_group.radii
        thicknesses = [
            self.surface_group.get_thickness(surf_idx)[0]
            for surf_idx in range(num_surfaces-1)
        ]

        # Scale radii & thicknesses
        for surf_idx in range(num_surfaces):
            if not np.isinf(radii[surf_idx]):
                self.set_radius(radii[surf_idx] * scale_factor, surf_idx)

            if (surf_idx != num_surfaces-1
               and not np.isinf(thicknesses[surf_idx])):
                self.set_thickness(thicknesses[surf_idx] * scale_factor,
                                   surf_idx)

        # Scale aperture, if aperture type is EPD
        if self.aperture.ap_type == 'EPD':
            self.aperture.value *= scale_factor

        # Scale physical apertures
        for surface in self.surface_group.surfaces:
            if surface.aperture is not None:
                surface.aperture.scale(scale_factor)

    def draw(self, fields='all', wavelengths='primary', num_rays=3,
             figsize=(10, 4), xlim=None, ylim=None):
        """
        Draw a 2D representation of the optical system.

        Args:
            fields (str or list, optional): The fields to be displayed.
                Defaults to 'all'.
            wavelengths (str or list, optional): The wavelengths to be
                displayed. Defaults to 'primary'.
            num_rays (int, optional): The number of rays to be traced for each
                field and wavelength. Defaults to 3.
            figsize (tuple, optional): The size of the figure. Defaults to
                (10, 4).
            xlim (tuple, optional): The x-axis limits of the plot. Defaults to
                None.
            ylim (tuple, optional): The y-axis limits of the plot. Defaults to
                None.
        """
        viewer = OpticViewer(self)
        viewer.view(fields, wavelengths, num_rays, distribution='line_y',
                    figsize=figsize, xlim=xlim, ylim=ylim)

    def draw3D(self, fields='all', wavelengths='primary', num_rays=24,
               figsize=(1200, 800), dark_mode=False):
        """
        Draw a 3D representation of the optical system.

        Args:
            fields (str or list, optional): The fields to be displayed.
                Defaults to 'all'.
            wavelengths (str or list, optional): The wavelengths to be
                displayed. Defaults to 'primary'.
            num_rays (int, optional): The number of rays to be traced for each
                field and wavelength. Defaults to 2.
            figsize (tuple, optional): The size of the figure. Defaults to
                (1200, 800).
            dark_mode (bool, optional): Whether to use dark mode. Defaults to
                False.
        """
        viewer = OpticViewer3D(self)
        viewer.view(fields, wavelengths, num_rays,
                    distribution='ring', figsize=figsize, dark_mode=dark_mode)

    def info(self):
        """Display the optical system information."""
        viewer = LensInfoViewer(self)
        viewer.view()

    def reset(self):
        """
        Reset the optical system to its initial state.
        """
        self.aperture = None
        self.field_type = None

        self.surface_group = SurfaceGroup()
        self.fields = FieldGroup()
        self.wavelengths = WavelengthGroup()

        self.paraxial = Paraxial(self)
        self.aberrations = Aberrations(self)
        self.ray_generator = RayGenerator(self)

        self.polarization = 'ignore'

        self.pickups = PickupManager(self)
        self.solves = SolveManager(self)
        self.obj_space_telecentric = False

    def n(self, wavelength='primary'):
        """
        Get the refractive indices of the surfaces.

        Args:
            wavelength (float or str, optional): The wavelength for which to
                calculate the refractive indices. Defaults to 'primary'.

        Returns:
            numpy.ndarray: The refractive indices of the surfaces.
        """
        if wavelength == 'primary':
            wavelength = self.primary_wavelength
        n = []
        for surface in self.surface_group.surfaces:
            n.append(surface.material_post.n(wavelength))
        return np.array(n)

    def update_paraxial(self):
        """
        Update the semi-aperture of the surfaces based on the paraxial
        analysis.
        """
        ya, _ = self.paraxial.marginal_ray()
        yb, _ = self.paraxial.chief_ray()
        ya = np.abs(np.ravel(ya))
        yb = np.abs(np.ravel(yb))
        for k, surface in enumerate(self.surface_group.surfaces):
            surface.set_semi_aperture(r_max=ya[k]+yb[k])

    def update(self):
        """
        Update the surfaces based on the pickup operations.
        """
        self.pickups.apply()
        self.solves.apply()

    def image_solve(self):
        """Update the image position such that the marginal ray crosses the
        optical axis at the image location."""
        ya, ua = self.paraxial.marginal_ray()
        offset = float(ya[-1, 0] / ua[-1, 0])
        self.surface_group.surfaces[-1].geometry.cs.z -= offset

    def trace(self, Hx, Hy, wavelength, num_rays=100,
              distribution='hexapolar'):
        """
        Trace a distribution of rays through the optical system.

        Args:
            Hx (float or numpy.ndarray): The normalized x field coordinate.
            Hy (float or numpy.ndarray): The normalized y field coordinate.
            wavelength (float): The wavelength of the rays.
            num_rays (int, optional): The number of rays to be traced. Defaults
                to 100.
            distribution (str or Distribution, optional): The distribution of
                the rays. Defaults to 'hexapolar'.

        Returns:
            RealRays: The RealRays object containing the traced rays.
        """

        if isinstance(distribution, str):
            distribution = create_distribution(distribution)
            distribution.generate_points(num_rays)
        Px = distribution.x
        Py = distribution.y

        rays = self.ray_generator.generate_rays(Hx, Hy, Px, Py, wavelength)
        self.surface_group.trace(rays)

        if isinstance(rays, PolarizedRays):
            rays.update_intensity(self.polarization_state)

        # update ray intensity
        self.surface_group.intensity[-1, :] = rays.i

        return rays

    def trace_generic(self, Hx, Hy, Px, Py, wavelength):
        """
        Trace generic rays through the optical system.

        Args:
            Hx (float or numpy.ndarray): The normalized x field coordinate.
            Hy (float or numpy.ndarray): The normalized y field coordinate.
            Px (float or numpy.ndarray): The normalized x pupil coordinate.
            Py (float or numpy.ndarray): The normalized y pupil coordinate
            wavelength (float): The wavelength of the rays.
        """
        vx, vy = self.fields.get_vig_factor(Hx, Hy)

        Px *= (1 - vx)
        Py *= (1 - vy)

        # assure all variables are arrays of the same size
        max_size = max([np.size(arr) for arr in [Hx, Hy, Px, Py]])
        Hx, Hy, Px, Py = [
            np.full(max_size, value) if isinstance(value, (float, int))
            else value if isinstance(value, np.ndarray)
            else None
            for value in [Hx, Hy, Px, Py]
        ]

        rays = self.ray_generator.generate_rays(Hx, Hy, Px, Py, wavelength)
        rays = self.surface_group.trace(rays)

        # update intensity
        self.surface_group.intensity[-1, :] = rays.i

        return rays

    def to_dict(self):
        """
        Convert the optical system to a dictionary.

        Returns:
            dict: The dictionary representation of the optical system.
        """
        data = {
            'version': 1.0,
            'aperture': self.aperture.to_dict() if self.aperture else None,
            'fields': self.fields.to_dict(),
            'wavelengths': self.wavelengths.to_dict(),
            'pickups': self.pickups.to_dict(),
            'solves': self.solves.to_dict(),
            'surface_group': self.surface_group.to_dict()
        }

        data['wavelengths']['polarization'] = self.polarization
        data['fields']['field_type'] = self.field_type
        data['fields']['object_space_telecentric'] = self.obj_space_telecentric
        return data

    @classmethod
    def from_dict(cls, data):
        """
        Create an optical system from a dictionary.

        Args:
            data (dict): The dictionary representation of the optical system.

        Returns:
            Optic: The optical system.
        """
        optic = cls()
        optic.aperture = Aperture.from_dict(data['aperture'])
        optic.surface_group = SurfaceGroup.from_dict(data['surface_group'])
        optic.fields = FieldGroup.from_dict(data['fields'])
        optic.wavelengths = WavelengthGroup.from_dict(data['wavelengths'])
        optic.pickups = PickupManager.from_dict(optic, data['pickups'])
        optic.solves = SolveManager.from_dict(optic, data['solves'])

        optic.polarization = data['wavelengths']['polarization']
        optic.field_type = data['fields']['field_type']
        optic.obj_space_telecentric = \
            data['fields']['object_space_telecentric']

        optic.paraxial = Paraxial(optic)
        optic.aberrations = Aberrations(optic)
        optic.ray_generator = RayGenerator(optic)

        return optic
