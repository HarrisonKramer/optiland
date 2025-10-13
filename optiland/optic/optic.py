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

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal

from optiland.aberrations import Aberrations
from optiland.aperture import Aperture
from optiland.apodization import BaseApodization
from optiland.fields import (
    AngleField,
    BaseFieldDefinition,
    Field,
    FieldGroup,
    ObjectHeightField,
    ParaxialImageHeightField,
)
from optiland.optic.optic_updater import OpticUpdater
from optiland.paraxial import Paraxial
from optiland.pickup import PickupManager
from optiland.rays import PolarizationState
from optiland.raytrace.real_ray_tracer import RealRayTracer
from optiland.solves import SolveManager
from optiland.surfaces import ObjectSurface, SurfaceGroup
from optiland.visualization import (
    LensInfoViewer,
    OpticViewer,
    OpticViewer3D,
    SurfaceSagViewer,
)
from optiland.wavelength import WavelengthGroup

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland._types import (
        BEArray,
        DistributionType,
        FieldType,
        ReferenceRay,
        ScalarOrArray,
        SurfaceParameters,
        SurfaceType,
        Unpack,
        WavelengthUnit,
    )
    from optiland.distribution import BaseDistribution
    from optiland.materials.base import BaseMaterial
    from optiland.rays import RealRays
    from optiland.surfaces.standard_surface import Surface


class Optic:
    """A class for defining and manipulating optical systems.

    The Optic class is central to the Optiland library, providing a comprehensive
    framework for representing optical systems. It encapsulates essential
    properties such as aperture, fields of view, constituent surfaces, and
    operating wavelengths. This class also offers a suite of methods for
    visualizing the optical system, performing ray tracing, and conducting
    paraxial and aberration analyses. Instances of Optic are fundamental objects
    passed to various analysis, optimization, and visualization functions
    throughout Optiland.

    Args:
        name (str, optional): An optional name for the optical system.
            Defaults to None.

    Attributes:
        name (str | None): An optional name for the optical system.
        aperture (Aperture | None): The aperture of the optical system.
        field_definition (BaseFieldDefinition | None): The definition of the field used
            in the optical system, e.g., AngleField or ObjectHeightField.
        surface_group (SurfaceGroup): The group of surfaces that constitute
            the optical system.
        fields (FieldGroup): The group of fields defined for the system.
        wavelengths (WavelengthGroup): The group of wavelengths used for
            analysis.
        paraxial (Paraxial): A helper class for paraxial analysis of the
            optical system.
        aberrations (Aberrations): A helper class for analyzing aberrations.
        ray_tracer (RealRayTracer): The ray tracer for performing real ray
            traces.
        polarization (PolarizationState | Literal['ignore']): The polarization
            state of the light. Defaults to 'ignore'.
        apodization (BaseApodization | None): The apodization function applied
            to the entrance pupil.
        pickups (PickupManager): Manages pickups, which link properties of one
            surface to another.
        solves (SolveManager): Manages solves, which automatically adjust
            surface properties to meet certain constraints.
        obj_space_telecentric (bool): If True, the system is object-space
            telecentric. Defaults to False.

    """

    def __init__(self, name: str | None = None):
        """
        Initializes an Optic instance.

        Args:
            name: An optional name for the optical system.
                Defaults to None.
        """
        self.name = name
        self.reset()

    def _initialize_attributes(self):
        """Initialize the attributes of the optical system."""
        self.aperture: Aperture | None = None
        self.field_definition: BaseFieldDefinition | None = None

        self.surface_group: SurfaceGroup = SurfaceGroup()
        self.fields: FieldGroup = FieldGroup()
        self.wavelengths: WavelengthGroup = WavelengthGroup()

        self.paraxial: Paraxial = Paraxial(self)
        self.aberrations: Aberrations = Aberrations(self)
        self.ray_tracer: RealRayTracer = RealRayTracer(self)

        self.polarization: PolarizationState | Literal["ignore"] = "ignore"

        self.apodization: BaseApodization | None = None
        self.pickups: PickupManager = PickupManager(self)
        self.solves: SolveManager = SolveManager(self)
        self.obj_space_telecentric: bool = False
        self._updater = OpticUpdater(self)

    def __add__(self, other: Optic) -> Optic:
        """Add two Optic objects together.

        This method combines the surfaces of the two Optic objects. The
        properties of the first optic (aperture, fields, etc.) are retained.

        Args:
            other (Optic): The Optic object to add to the current one.

        Returns:
            Optic: A new Optic object containing the combined surfaces.
        """
        new_optic = deepcopy(self)
        new_optic.surface_group += other.surface_group
        return new_optic

    @property
    def primary_wavelength(self) -> float:
        """The primary wavelength in microns."""
        return self.wavelengths.primary_wavelength.value

    @property
    def object_surface(self) -> ObjectSurface | None:
        """The object surface instance (`ObjectSurface` or `None`)."""
        for surface in self.surface_group.surfaces:
            if isinstance(surface, ObjectSurface):
                return surface
        return None

    @property
    def image_surface(self) -> Surface:
        """The image surface instance."""
        return self.surface_group.surfaces[-1]

    @property
    def total_track(self) -> float:
        """The total track length of the system."""
        return self.surface_group.total_track

    @property
    def polarization_state(self) -> PolarizationState | None:
        """The polarization state of the optic.

        Returns:
            PolarizationState | None: The `PolarizationState` object if
            polarization is considered, otherwise `None`.

        Raises:
            ValueError: If the polarization state is invalid.
        """
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
        new_surface: Surface | None = None,
        surface_type: SurfaceType = "standard",
        comment: str = "",
        index: int | None = None,
        is_stop: bool = False,
        material: str | BaseMaterial = "air",
        **kwargs: Unpack[SurfaceParameters],
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
            ValueError: If a new surface is provided and no index is given.
            IndexError: If the index is out of bounds for insertion, or negative.

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

    def add_field(self, y: float, x: float = 0.0, vx: float = 0.0, vy: float = 0.0):
        """Add a field to the optical system.

        Args:
            y: The y-coordinate of the field.
            x: The x-coordinate of the field.
                Defaults to 0.0.
            vx: The x-component of the field's vignetting
                factor. Defaults to 0.0.
            vy: The y-component of the field's vignetting
                factor. Defaults to 0.0.

        """
        new_field = Field(x, y, vx, vy)
        self.fields.add_field(new_field)

    def add_wavelength(
        self,
        value: float,
        is_primary: bool = False,
        unit: WavelengthUnit = "um",
        weight: float = 1.0,
    ):
        """Add a wavelength to the optical system.

        Args:
            value (float): The value of the wavelength.
            is_primary (bool, optional): If True, this wavelength is set as the
                primary wavelength. Defaults to False.
            unit (WavelengthUnit, optional): The unit of the wavelength.
                Defaults to 'um'.
            weight (float, optional): The weight of the wavelength for
                polychromatic analysis. Defaults to 1.0.

        """
        self.wavelengths.add_wavelength(
            value=value, is_primary=is_primary, unit=unit, weight=weight
        )

    def set_aperture(self, aperture_type: str, value: float):
        """Set the aperture of the optical system.

        Args:
            aperture_type (str): The type of the aperture. Must be one of 'EPD',
                'imageFNO', or 'objectNA'.
            value (float): The value of the aperture.

        """
        self.aperture = Aperture(aperture_type, value)

    def set_field_type(self, field_type: FieldType):
        """Set the type of field used in the optical system.

        Args:
            field_type (FieldType): The type of field, e.g., 'angle',
                'object_height', or 'paraxial_image_height'.

        Raises:
            ValueError: If the field type is invalid.
        """
        if field_type == "angle":
            self.field_definition = AngleField()
        elif field_type == "object_height":
            self.field_definition = ObjectHeightField()
        elif field_type == "paraxial_image_height":
            self.field_definition = ParaxialImageHeightField()
        else:
            raise ValueError(f"Invalid field type: {field_type}.")

    def set_radius(self, value: float, surface_number: int):
        """Set the radius of curvature of a surface.

        Args:
            value (float): The value of the radius.
            surface_number (int): The index of the surface.

        """
        self._updater.set_radius(value, surface_number)

    def set_conic(self, value: float, surface_number: int):
        """Set the conic constant of a surface.

        Args:
            value (float): The value of the conic constant.
            surface_number (int): The index of the surface.

        """
        self._updater.set_conic(value, surface_number)

    def set_thickness(self, value: float, surface_number: int):
        """Set the thickness of a surface.

        Args:
            value (float): The value of the thickness.
            surface_number (int): The index of the surface.

        """
        self._updater.set_thickness(value, surface_number)

    def set_index(self, value: float, surface_number: int):
        """Set the index of refraction of a surface.

        Args:
            value (float): The value of the index of refraction.
            surface_number (int): The index of the surface.

        """
        self._updater.set_index(value, surface_number)

    def set_material(self, material: BaseMaterial, surface_number: int):
        """Set the material of a surface.

        Args:
            material (BaseMaterial): The material.
            surface_number (int): The index of the surface.

        """
        self._updater.set_material(material, surface_number)

    def set_norm_radius(self, value: float, surface_number: int):
        """Set the normalization radius of a surface.

        Args:
            value (float): The value of the normalization radius.
            surface_number (int): The index of the surface.
        """
        self._updater.set_norm_radius(value, surface_number)

    def set_asphere_coeff(
        self, value: float, surface_number: int, aspher_coeff_idx: int
    ):
        """Set an aspheric coefficient on a surface.

        Args:
            value (float): The value of the aspheric coefficient.
            surface_number (int): The index of the surface.
            aspher_coeff_idx (int): The index of the aspheric coefficient to
                set.

        """
        self._updater.set_asphere_coeff(value, surface_number, aspher_coeff_idx)

    def set_polarization(self, polarization: PolarizationState | Literal["ignore"]):
        """Set the polarization state of the optic.

        Args:
            polarization (PolarizationState | Literal['ignore']): The polarization
                state to set. It can be either a `PolarizationState` object or
                'ignore'.

        """
        self._updater.set_polarization(polarization)

    def set_apodization(self, apodization: BaseApodization):
        """Set the apodization of the optical system.

        Args:
            apodization (BaseApodization): The apodization object to set.
        """
        self._updater.set_apodization(apodization)

    def scale_system(self, scale_factor: float):
        """Scales the optical system by a given scale factor.

        Args:
            scale_factor (float): The factor by which to scale the system.

        """
        self._updater.scale_system(scale_factor)

    def update_paraxial(self):
        """Update the semi-aperture of the surfaces based on paraxial analysis."""
        self._updater.update_paraxial()

    def update_normalization(self, surface: Surface) -> None:
        """Update the normalization radius of surfaces."""
        self._updater.update_normalization(surface)

    def update(self) -> None:
        """Update the surface properties (pickups, solves, paraxial properties)."""
        self._updater.update()

    def image_solve(self):
        """Update the image position such that the marginal ray crosses the optical axis
        at the image location.
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
        fields: list[tuple[float, float]] | Literal["all"] = "all",
        wavelengths: list[float] | Literal["primary"] = "primary",
        num_rays: int = 3,
        distribution: DistributionType = "line_y",
        figsize: tuple[float, float] = (10, 4),
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        title: str | None = None,
        reference: ReferenceRay | None = None,
    ) -> tuple[Figure, Axes]:
        """Draw a 2D representation of the optical system.

        Args:
            fields (list[tuple[float, float]] | Literal['all'], optional): The fields to
                be displayed, specified by their indices. Defaults to 'all'.
            wavelengths (list[float] | Literal['primary'], optional): The
                wavelengths to be displayed, specified by their indices.
                Defaults to 'primary'.
            num_rays (int, optional): The number of rays to trace for each
                field and wavelength. Defaults to 3.
            distribution (DistributionType, optional): The distribution of
                rays to trace. Defaults to 'line_y'.
            figsize (tuple[float, float], optional): The size of the figure.
                Defaults to (10, 4).
            xlim (tuple[float, float] | None, optional): The x-axis limits of
                the plot. Defaults to None.
            ylim (tuple[float, float] | None, optional): The y-axis limits of
                the plot. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to
                None.
            reference (ReferenceRay | None, optional): The reference rays to
                plot, e.g., 'chief' or 'marginal'. Defaults to None.

        Returns:
            tuple[Figure, Axes]: A tuple containing the matplotlib Figure and
            Axes objects of the plot.

        """
        viewer = OpticViewer(self)
        fig, ax = viewer.view(
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
        return fig, ax

    def draw3D(
        self,
        fields: list[tuple[float, float]] | Literal["all"] = "all",
        wavelengths: list[float] | Literal["primary"] = "primary",
        num_rays: int = 24,
        distribution: DistributionType = "ring",
        figsize: tuple[float, float] = (1200, 800),
        dark_mode: bool = False,
        reference: ReferenceRay | None = None,
    ):
        """Draw a 3D representation of the optical system.

        Args:
            fields (list[tuple[float, float]] | Literal['all'], optional): The fields to
                be displayed, specified by their indices. Defaults to 'all'.
            wavelengths (list[int] | Literal['primary'], optional): The
                wavelengths to be displayed, specified by their indices.
                Defaults to 'primary'.
            num_rays (int, optional): The number of rays to trace for each
                field and wavelength. Defaults to 24.
            distribution (DistributionType, optional): The distribution of
                rays to trace. Defaults to 'ring'.
            figsize (tuple[float, float], optional): The size of the figure.
                Defaults to (1200, 800).
            dark_mode (bool, optional): If True, use a dark theme for the
                plot. Defaults to False.
            reference (ReferenceRay | None, optional): The reference rays to
                plot, e.g., 'chief' or 'marginal'. Defaults to None.

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

    def n(self, wavelength: float | Literal["primary"] = "primary") -> BEArray:
        """Get the refractive indices of materials at a given wavelength.

        This method calculates the refractive indices for each space between
        surfaces in the optical system.

        Args:
            wavelength (float | Literal['primary'], optional): The wavelength
                in microns for which to calculate the refractive indices.
                Can be a float value or 'primary' to use the system's
                primary wavelength. Defaults to 'primary'.

        Returns:
            be.ndarray: An array of refractive indices for each space.

        """
        if wavelength == "primary":
            wavelength = self.primary_wavelength

        return self.surface_group.n(wavelength)

    def trace(
        self,
        Hx: ScalarOrArray,
        Hy: ScalarOrArray,
        wavelength: float,
        num_rays: int | None = 100,
        distribution: DistributionType | BaseDistribution | None = "hexapolar",
    ) -> RealRays:
        """Trace a distribution of rays through the optical system.

        Args:
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            wavelength (float): The wavelength of the rays in microns.
            num_rays: The number of rays to trace.
                Defaults to 100.
            distribution:
                The distribution of rays. Can be a string identifier (e.g.,
                'hexapolar', 'uniform') or a `BaseDistribution` object.
                Defaults to 'hexapolar'.

        Returns:
            RealRays: A `RealRays` object containing the traced rays.

        """
        return self.ray_tracer.trace(Hx, Hy, wavelength, num_rays, distribution)

    def trace_generic(
        self,
        Hx: ScalarOrArray,
        Hy: ScalarOrArray,
        Px: ScalarOrArray,
        Py: ScalarOrArray,
        wavelength: float,
    ):
        """Trace generic rays through the optical system.

        Args:
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            Px: The normalized x pupil coordinate(s).
            Py: The normalized y pupil coordinate(s).
            wavelength (float): The wavelength of the rays in microns.

        Returns:
            RealRays: A `RealRays` object containing the traced rays.

        """
        return self.ray_tracer.trace_generic(Hx, Hy, Px, Py, wavelength)

    def plot_surface_sag(
        self, surface_index: int, y_cross_section: float = 0, x_cross_section: float = 0
    ):
        """Analyzes and visualizes the sag of a given lens surface.

        Args:
            surface_index: The index of the surface to analyze.
            y_cross_section: The y-coordinate for the
                x-sag plot. Defaults to 0.
            x_cross_section: The x-coordinate for the
                y-sag plot. Defaults to 0.
        """
        viewer = SurfaceSagViewer(self)
        viewer.view(surface_index, y_cross_section, x_cross_section)

    def to_dict(self) -> dict:
        """Convert the optical system to a dictionary.

        Returns:
            The dictionary representation of the optical system.

        """
        data = {
            "version": 1.0,
            "aperture": self.aperture.to_dict() if self.aperture else None,
            "fields": self.fields.to_dict(),
            "wavelengths": self.wavelengths.to_dict(),
            "apodization": self.apodization.to_dict() if self.apodization else None,
            "pickups": self.pickups.to_dict(),
            "solves": self.solves.to_dict(),
            "surface_group": self.surface_group.to_dict(),
        }

        data["wavelengths"]["polarization"] = self.polarization
        data["fields"]["field_definition"] = (
            self.field_definition.to_dict() if self.field_definition else None
        )
        data["fields"]["object_space_telecentric"] = self.obj_space_telecentric
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Optic:
        """Create an optical system from a dictionary.

        Args:
            data: The dictionary representation of the optical system.

        Returns:
            The optical system.

        """
        optic = cls()
        optic.aperture = Aperture.from_dict(data["aperture"])
        optic.surface_group = SurfaceGroup.from_dict(data["surface_group"])
        optic.fields = FieldGroup.from_dict(data["fields"])
        optic.wavelengths = WavelengthGroup.from_dict(data["wavelengths"])

        apodization_data = data.get("apodization")
        if apodization_data:
            optic.apodization = BaseApodization.from_dict(apodization_data)

        optic.pickups = PickupManager.from_dict(optic, data["pickups"])
        optic.solves = SolveManager.from_dict(optic, data["solves"])

        optic.polarization = data["wavelengths"]["polarization"]
        if data["fields"].get("field_definition"):
            optic.field_definition = BaseFieldDefinition.from_dict(
                data["fields"]["field_definition"]
            )
        elif data["fields"].get("field_type"):
            optic.set_field_type(data["fields"]["field_type"])
        else:
            optic.field_definition = None
        optic.obj_space_telecentric = data["fields"]["object_space_telecentric"]

        optic.paraxial = Paraxial(optic)
        optic.aberrations = Aberrations(optic)
        optic.ray_tracer = RealRayTracer(optic)

        return optic
