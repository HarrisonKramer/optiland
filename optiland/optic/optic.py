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

from copy import deepcopy
from typing import Union

from optiland.aberrations import Aberrations
from optiland.aperture import Aperture
from optiland.fields import Field, FieldGroup
from optiland.optic.optic_updater import OpticUpdater
from optiland.paraxial import Paraxial
from optiland.pickup import PickupManager
from optiland.rays import PolarizationState, RayGenerator
from optiland.raytrace.real_ray_tracer import RealRayTracer
from optiland.solves import SolveManager
from optiland.surfaces import ObjectSurface, SurfaceGroup
from optiland.visualization import LensInfoViewer, OpticViewer, OpticViewer3D
from optiland.wavelength import WavelengthGroup


class Optic:
    """The Optic class represents an optical system.

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

    def __init__(self, name: str = None):
        self.name = name
        self.reset()

    def _initialize_attributes(self):
        """Reset the optical system to its initial state."""
        self.aperture = None
        self.field_type = None

        self.surface_group = SurfaceGroup()
        self.fields = FieldGroup()
        self.wavelengths = WavelengthGroup()

        self.paraxial = Paraxial(self)
        self.aberrations = Aberrations(self)
        self.ray_tracer = RealRayTracer(self)

        self.polarization = "ignore"

        self.pickups = PickupManager(self)
        self.solves = SolveManager(self)
        self.obj_space_telecentric = False
        self._updater = OpticUpdater(self)

    def __add__(self, other):
        """Add two Optic objects together."""
        new_optic = deepcopy(self)
        new_optic.surface_group += other.surface_group
        return new_optic

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
        return self.surface_group.total_track

    @property
    def polarization_state(self):
        """PolarizationState: the polarization state of the optic"""
        if self.polarization == "ignore":
            return None
        elif isinstance(self.polarization, PolarizationState):
            return self.polarization
        else:
            raise ValueError(
                "Invalid polarization state. Must be either "
                'PolarizationState or "ignore".',
            )

    def reset(self):
        """Reset the optical system to its initial state."""
        self._initialize_attributes()

    def add_surface(
        self,
        new_surface=None,
        surface_type="standard",
        comment="",
        index=None,
        is_stop=False,
        material="air",
        **kwargs,
    ):
        """Adds a new surface to the optic.

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
            new_surface=new_surface,
            surface_type=surface_type,
            comment=comment,
            index=index,
            is_stop=is_stop,
            material=material,
            **kwargs,
        )

    def add_field(self, y, x=0.0, vx=0.0, vy=0.0):
        """Add a field to the optical system.

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

    def add_wavelength(self, value, is_primary=False, unit="um"):
        """Add a wavelength to the optical system.

        Args:
            value (float): The value of the wavelength.
            is_primary (bool, optional): Whether the wavelength is the primary
                wavelength. Defaults to False.
            unit (str, optional): The unit of the wavelength. Defaults to 'um'.

        """
        self.wavelengths.add_wavelength(value, is_primary, unit)

    def set_aperture(self, aperture_type, value):
        """Set the aperture of the optical system.

        Args:
            aperture_type (str): The type of the aperture.
            value (float): The value of the aperture.

        """
        self.aperture = Aperture(aperture_type, value)

    def set_field_type(self, field_type):
        """Set the type of field used in the optical system.

        Args:
            field_type (str): The type of field.

        """
        if field_type not in ["angle", "object_height"]:
            raise ValueError('Invalid field type. Must be "angle" or "object_height".')
        self.field_type = field_type

    def set_radius(self, value, surface_number):
        """Set the radius of curvature of a surface.

        Args:
            value (float): The value of the radius.
            surface_number (int): The index of the surface.

        """
        self._updater.set_radius(value, surface_number)

    def set_conic(self, value, surface_number):
        """Set the conic constant of a surface.

        Args:
            value (float): The value of the conic constant.
            surface_number (int): The index of the surface.

        """
        self._updater.set_conic(value, surface_number)

    def set_thickness(self, value, surface_number):
        """Set the thickness of a surface.

        Args:
            value (float): The value of the thickness.
            surface_number (int): The index of the surface.

        """
        self._updater.set_thickness(value, surface_number)

    def set_index(self, value, surface_number):
        """Set the index of refraction of a surface.

        Args:
            value (float): The value of the index of refraction.
            surface_number (int): The index of the surface.

        """
        self._updater.set_index(value, surface_number)

    def set_asphere_coeff(self, value, surface_number, aspher_coeff_idx):
        """Set the asphere coefficient on a surface

        Args:
            value (float): The value of aspheric coefficient
            surface_number (int): The index of the surface.
            aspher_coeff_idx (int): index of the aspheric coefficient on the
                surface

        """
        self._updater.set_asphere_coeff(value, surface_number, aspher_coeff_idx)

    def set_polarization(self, polarization: Union[PolarizationState, str]):
        """Set the polarization state of the optic.

        Args:
            polarization (Union[PolarizationState, str]): The polarization
                state to set. It can be either a `PolarizationState` object or
                'ignore'.

        """
        self._updater.set_polarization(polarization)

    def scale_system(self, scale_factor):
        """Scales the optical system by a given scale factor.

        Args:
            scale_factor (float): The factor by which to scale the system.

        """
        self._updater.scale_system(scale_factor)

    def update_paraxial(self):
        """Update the semi-aperture of the surfaces based on the paraxial
        analysis.
        """
        self._updater.update_paraxial()

    def update_normalization(self, surface) -> None:
        """Update the normalization radius of non-spherical surfaces."""
        self._updater.update_normalization(surface)

    def update(self) -> None:
        """Update the surface properties (pickups, solves, paraxial properties)."""
        self._updater.update()

    def image_solve(self):
        """Update the image position such that the marginal ray crosses the
        optical axis at the image location.
        """
        self._updater.image_solve()

    def flip(self):
        """Flips the optical system.

        This reverses the order of surfaces (excluding object and image planes),
        their geometries, and materials. Pickups and solves referencing surface
        indices are updated accordingly. The coordinate system is adjusted such
        that the new first optical surface (originally the last one in the
        flipped segment) is placed at z=0.0.
        """
        self._updater.flip()

    def draw(
        self,
        fields="all",
        wavelengths="primary",
        num_rays=3,
        distribution="line_y",
        figsize=(10, 4),
        xlim=None,
        ylim=None,
        title=None,
        reference=None,
    ):
        """Draw a 2D representation of the optical system.

        Args:
            fields (str or list, optional): The fields to be displayed.
                Defaults to 'all'.
            wavelengths (str or list, optional): The wavelengths to be
                displayed. Defaults to 'primary'.
            num_rays (int, optional): The number of rays to be traced for each
                field and wavelength. Defaults to 3.
            distribution (str, optional): The distribution of the rays.
                Defaults to 'line_y'.
            figsize (tuple, optional): The size of the figure. Defaults to
                (10, 4).
            xlim (tuple, optional): The x-axis limits of the plot. Defaults to
                None.
            ylim (tuple, optional): The y-axis limits of the plot. Defaults to
                None.
            reference (str, optional): The reference rays to plot. Options
                include "chief" and "marginal". Defaults to None.

        """
        viewer = OpticViewer(self)
        viewer.view(
            fields,
            wavelengths,
            num_rays,
            distribution=distribution,
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
            title=title,
            reference=reference,
        )

    def draw3D(
        self,
        fields="all",
        wavelengths="primary",
        num_rays=24,
        distribution="ring",
        figsize=(1200, 800),
        dark_mode=False,
        reference=None,
    ):
        """Draw a 3D representation of the optical system.

        Args:
            fields (str or list, optional): The fields to be displayed.
                Defaults to 'all'.
            wavelengths (str or list, optional): The wavelengths to be
                displayed. Defaults to 'primary'.
            num_rays (int, optional): The number of rays to be traced for each
                field and wavelength. Defaults to 2.
            distribution (str, optional): The distribution of the rays.
                Defaults to 'ring'.
            figsize (tuple, optional): The size of the figure. Defaults to
                (1200, 800).
            dark_mode (bool, optional): Whether to use dark mode. Defaults to
                False.
            reference (str, optional): The reference rays to plot. Options
                include "chief" and "marginal". Defaults to None.

        """
        viewer = OpticViewer3D(self)
        viewer.view(
            fields,
            wavelengths,
            num_rays,
            distribution=distribution,
            figsize=figsize,
            dark_mode=dark_mode,
            reference=reference,
        )

    def info(self):
        """Display the optical system information."""
        viewer = LensInfoViewer(self)
        viewer.view()

    def n(self, wavelength: Union[float, str] = "primary"):
        """Get the refractive indices of the materials for each space between
        surfaces at a given wavelength.

        Args:
            wavelength (float or str, optional): The wavelength in microns for
                which to calculate the refractive indices. Can be a float value
                or the string 'primary' to use the system's primary wavelength.
                Defaults to 'primary'.

        Returns:
            be.ndarray: An array of refractive indices for each space.

        """
        if wavelength == "primary":
            wavelength = self.primary_wavelength
        return self.surface_group.n(wavelength)

    def trace(self, Hx, Hy, wavelength, num_rays=100, distribution="hexapolar"):
        """Trace a distribution of rays through the optical system.

        Args:
            Hx (float or be.ndarray): The normalized x field coordinate(s).
            Hy (float or be.ndarray): The normalized y field coordinate(s).
            wavelength (float): The wavelength of the rays in microns.
            num_rays (int, optional): The number of rays to be traced.
                Defaults to 100.
            distribution (str or optiland.distribution.BaseDistribution, optional):
                The distribution of the rays. Can be a string identifier (e.g.,
                'hexapolar', 'uniform') or a Distribution object.
                Defaults to 'hexapolar'.

        Returns:
            RealRays: The RealRays object containing the traced rays.

        """
        return self.ray_tracer.trace(Hx, Hy, wavelength, num_rays, distribution)

    def trace_generic(self, Hx, Hy, Px, Py, wavelength):
        """Trace generic rays through the optical system.

        Args:
            Hx (float or be.ndarray): The normalized x field coordinate(s).
            Hy (float or be.ndarray): The normalized y field coordinate(s).
            Px (float or be.ndarray): The normalized x pupil coordinate(s).
            Py (float or be.ndarray): The normalized y pupil coordinate(s).
            wavelength (float): The wavelength of the rays in microns.

        """
        return self.ray_tracer.trace_generic(Hx, Hy, Px, Py, wavelength)

    def to_dict(self):
        """Convert the optical system to a dictionary.

        Returns:
            dict: The dictionary representation of the optical system.

        """
        data = {
            "version": 1.0,
            "aperture": self.aperture.to_dict() if self.aperture else None,
            "fields": self.fields.to_dict(),
            "wavelengths": self.wavelengths.to_dict(),
            "pickups": self.pickups.to_dict(),
            "solves": self.solves.to_dict(),
            "surface_group": self.surface_group.to_dict(),
        }

        data["wavelengths"]["polarization"] = self.polarization
        data["fields"]["field_type"] = self.field_type
        data["fields"]["object_space_telecentric"] = self.obj_space_telecentric
        return data

    @classmethod
    def from_dict(cls, data):
        """Create an optical system from a dictionary.

        Args:
            data (dict): The dictionary representation of the optical system.

        Returns:
            Optic: The optical system.

        """
        optic = cls()
        optic.aperture = Aperture.from_dict(data["aperture"])
        optic.surface_group = SurfaceGroup.from_dict(data["surface_group"])
        optic.fields = FieldGroup.from_dict(data["fields"])
        optic.wavelengths = WavelengthGroup.from_dict(data["wavelengths"])
        optic.pickups = PickupManager.from_dict(optic, data["pickups"])
        optic.solves = SolveManager.from_dict(optic, data["solves"])

        optic.polarization = data["wavelengths"]["polarization"]
        optic.field_type = data["fields"]["field_type"]
        optic.obj_space_telecentric = data["fields"]["object_space_telecentric"]

        optic.paraxial = Paraxial(optic)
        optic.aberrations = Aberrations(optic)
        optic.ray_generator = RayGenerator(optic)

        return optic
