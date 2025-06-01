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

from optiland.aberrations import Aberrations # Used in _initialize_attributes
# Aperture will be used by OpticConfigurator
from optiland.fields import FieldGroup, FieldType # Field moved to OpticConfigurator, FieldType still used
from optiland.optic.optic_updater import OpticUpdater
from optiland.optic.optic_serializer import OpticSerializer
from optiland.paraxial import Paraxial # Used in _initialize_attributes
# PickupManager, SolveManager, are no longer directly instantiated or called in from_dict here.
# OpticSerializer handles their from_dict. OpticUpdater initializes them.
from optiland.rays import PolarizationState, RayGenerator, PolarizationType # Added PolarizationType, RayGenerator used by RealRayTracer
from optiland.raytrace.real_ray_tracer import RealRayTracer # Used in _initialize_attributes
# SolveManager is used by _initialize_attributes via OpticUpdater which might need it. Or pickups/solves are set there.
# Let's check _initialize_attributes: self.pickups = PickupManager(self), self.solves = SolveManager(self)
# So PickupManager and SolveManager are needed for _initialize_attributes.
from optiland.pickup import PickupManager # Needed for _initialize_attributes
from optiland.solves import SolveManager # Needed for _initialize_attributes
from optiland.surfaces import ObjectSurface, SurfaceGroup # SurfaceGroup used in _initialize_attributes, add_surface, type hints
from optiland.optic.optic_configurator import OpticConfigurator # Added
# Visualization imports moved to OpticVisualizer
from optiland.optic.optic_visualization import OpticVisualizer # Added import
from optiland.optic.optic_raytracer import OpticRayTracer # Added import
from optiland.wavelength import WavelengthGroup # WavelengthGroup used in _initialize_attributes, add_wavelength, type hints


class Optic:
    """The Optic class represents an optical system.

    Attributes:
        aperture (Aperture): The aperture of the optical system.
        field_type (FieldType): The type of field used in the optical system.
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
        self.field_type: FieldType = None # Explicitly None, to be set by set_field_type

        self.surface_group = SurfaceGroup()
        self.fields = FieldGroup()
        self.wavelengths = WavelengthGroup()

        self.paraxial = Paraxial(self)
        self.aberrations = Aberrations(self)
        self.ray_tracer = RealRayTracer(self)

        self.polarization: Union[PolarizationType, PolarizationState] = PolarizationType.IGNORE # Default to Enum

        self.pickups = PickupManager(self) # Initialized here
        self.solves = SolveManager(self)   # Initialized here
        self.obj_space_telecentric = False
        self._updater = OpticUpdater(self)
        self._serializer = OpticSerializer(self) # Added serializer
        self._visualizer = OpticVisualizer(self) # Added visualizer
        self._raytracer_facade = OpticRayTracer(self) # Added raytracer facade
        self._configurator = OpticConfigurator(self) # Added configurator

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
        """PolarizationState: the polarization state of the optic, or None if polarization is ignored."""
        if self.polarization == PolarizationType.IGNORE:
            return None
        elif isinstance(self.polarization, PolarizationState):
            return self.polarization
        # Add handling for other PolarizationType members if they don't directly map to a PolarizationState object
        # For now, if it's another Enum member (e.g. JONES_VECTOR that isn't a full PolarizationState object yet)
        # it might also mean no specific state object is available, or a default one should be returned.
        # Based on current structure, only PolarizationState objects or IGNORE are primary states.
        # If self.polarization is an Enum member like JONES_VECTOR but not a full PolarizationState object,
        # this implies that a PolarizationState object needs to be constructed or is not fully defined.
        # For now, we assume if it's not IGNORE and not a PolarizationState object, it's an invalid state for this property.
        # However, the setter now ensures self.polarization is either PolarizationType.IGNORE or a PolarizationState object,
        # or another PolarizationType member.
        elif isinstance(self.polarization, PolarizationType) and self.polarization != PolarizationType.IGNORE:
            # This case needs clarification: if self.polarization is e.g. PolarizationType.JONES_VECTOR,
            # what should this property return? A default PolarizationState? For now, returning None.
            # This implies that to have a specific state, a PolarizationState object must be set.
            # Or, this property could raise an error if it's an enum type that's not IGNORE and not a state obj.
            # Given current logic, if it's not IGNORE, it should be a PolarizationState obj for this property to make sense.
            # The OpticUpdater.set_polarization ensures this.
            # The only enum types stored directly on self.optic.polarization by updater are PolarizationType.IGNORE
            # or other PolarizationType members if they were passed as such.
            # Let's assume if it's a PolarizationType enum member other than IGNORE, it means a specific state object is implied
            # but not yet fully represented as a PolarizationState object. For this property, it might be an error or None.
            # Given the original code, it only returned a PolarizationState or None.
            # The current set_polarization in OpticUpdater allows self.optic.polarization to be PolarizationType.JONES_VECTOR.
            # What should polarization_state return then?
            # For safety and consistency with original behavior (returns PolarizationState or None):
            return None # Or raise an error for unexpected enum members here.
        else:
            # This case should ideally not be reached if set_polarization works as expected.
            raise ValueError(
                f"Invalid internal polarization state: {self.polarization}. Expected PolarizationType.IGNORE or PolarizationState instance."
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
        return self._configurator.add_surface(
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
        return self._configurator.add_field(y, x=x, vx=vx, vy=vy)

    def add_wavelength(self, value, is_primary=False, unit="um"):
        """Add a wavelength to the optical system.

        Args:
            value (float): The value of the wavelength.
            is_primary (bool, optional): Whether the wavelength is the primary
                wavelength. Defaults to False.
            unit (str, optional): The unit of the wavelength. Defaults to 'um'.

        """
        return self._configurator.add_wavelength(value, is_primary=is_primary, unit=unit)

    def set_aperture(self, aperture_type, value):
        """Set the aperture of the optical system.

        Args:
            aperture_type (str): The type of the aperture.
            value (float): The value of the aperture.

        """
        return self._configurator.set_aperture(aperture_type, value)

    def set_field_type(self, field_type_input: Union[str, FieldType]):
        """Set the type of field used in the optical system.

        Args:
            field_type_input (Union[str, FieldType]): The type of field.
        """
        return self._configurator.set_field_type(field_type_input)

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

    def set_polarization(self, polarization_input: Union[PolarizationState, str, PolarizationType]):
        """Set the polarization state of the optic.

        Args:
            polarization_input (Union[PolarizationState, str, PolarizationType]):
                The polarization state to set. It can be a `PolarizationState` object,
                a string (e.g., "ignore"), or a `PolarizationType` enum member.
        """
        self._updater.set_polarization(polarization_input)

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
        self._visualizer.draw(
            fields=fields,
            wavelengths=wavelengths,
            num_rays=num_rays,
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
        self._visualizer.draw3D(
            fields=fields,
            wavelengths=wavelengths,
            num_rays=num_rays,
            distribution=distribution,
            figsize=figsize,
            dark_mode=dark_mode,
            reference=reference,
        )

    def info(self):
        """Display the optical system information."""
        self._visualizer.info()

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
        return self._raytracer_facade.trace(Hx, Hy, wavelength, num_rays, distribution)

    def trace_generic(self, Hx, Hy, Px, Py, wavelength):
        """Trace generic rays through the optical system.

        Args:
            Hx (float or be.ndarray): The normalized x field coordinate(s).
            Hy (float or be.ndarray): The normalized y field coordinate(s).
            Px (float or be.ndarray): The normalized x pupil coordinate(s).
            Py (float or be.ndarray): The normalized y pupil coordinate(s).
            wavelength (float): The wavelength of the rays in microns.
        """
        return self._raytracer_facade.trace_generic(Hx, Hy, Px, Py, wavelength)

    def to_dict(self):
        """Convert the optical system to a dictionary.

        Returns:
            dict: The dictionary representation of the optical system.

        """
        return self._serializer.to_dict()

    @classmethod
    def from_dict(cls, data):
        """Create an optical system from a dictionary.

        Args:
            data (dict): The dictionary representation of the optical system.

        Returns:
            Optic: The optical system.

        """
        return OpticSerializer.from_dict(data, cls=cls)
