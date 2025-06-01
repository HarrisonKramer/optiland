from typing import Union
from optiland.aperture import Aperture
from optiland.fields import Field, FieldType

class OpticConfigurator:
    def __init__(self, optic):
        self.optic = optic

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
        (Method moved from Optic class)
        """
        self.optic.surface_group.add_surface(
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
        (Method moved from Optic class)
        """
        # Ensure self.optic.field_type is not None before creating a Field
        if self.optic.field_type is None:
            raise ValueError("Cannot add field: Optic field_type is not set. Call set_field_type() first.")
        new_field = Field(self.optic.field_type, x, y, vx, vy)
        self.optic.fields.add_field(new_field)

    def add_wavelength(self, value, is_primary=False, unit="um"):
        """Add a wavelength to the optical system.
        (Method moved from Optic class)
        """
        self.optic.wavelengths.add_wavelength(value, is_primary, unit)

    def set_aperture(self, aperture_type, value):
        """Set the aperture of the optical system.
        (Method moved from Optic class)
        """
        self.optic.aperture = Aperture(aperture_type, value)

    def set_field_type(self, field_type_input: Union[str, FieldType]):
        """Set the type of field used in the optical system.
        (Method moved from Optic class)
        """
        if isinstance(field_type_input, str):
            try:
                self.optic.field_type = FieldType(field_type_input.lower())
            except ValueError: # Handles if the string is not a valid FieldType value
                raise ValueError(f"Invalid field_type string: '{field_type_input}'. Must be 'angle' or 'object_height'.")
        elif isinstance(field_type_input, FieldType):
            self.optic.field_type = field_type_input
        else:
            raise TypeError(f"field_type_input must be a string or FieldType enum, not {type(field_type_input)}.")
