"""Optic Serialization Module

This module provides functions for serializing and deserializing Optic objects.
It aims to keep the Optic class cleaner by separating serialization concerns.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.optic.optic import Optic

# New imports required by optic_from_dict
from optiland.aberrations import Aberrations
from optiland.aperture import Aperture
from optiland.fields import FieldGroup
from optiland.paraxial import Paraxial
from optiland.pickup import PickupManager
from optiland.rays import RayGenerator
from optiland.solves import SolveManager
from optiland.surfaces import SurfaceGroup
from optiland.wavelength import WavelengthGroup
# It's important that Optic is imported here for the Optic() call
# but also that it's available for type hinting.
# The TYPE_CHECKING block above handles the type hinting.
from optiland.optic.optic import Optic


def optic_to_dict(optic: 'Optic') -> dict:
    """Converts an Optic object to a dictionary representation.

    This function takes an Optic object and serializes its core attributes
    into a dictionary. This allows for saving, loading, or transmitting
    the optical system's definition.

    Args:
        optic: The Optic object to serialize.

    Returns:
        A dictionary containing the serialized representation of the Optic
        object.
    """
    data = {
        "version": 1.0,
        "aperture": optic.aperture.to_dict() if optic.aperture else None,
        "fields": optic.fields.to_dict(),
        "wavelengths": optic.wavelengths.to_dict(),
        "pickups": optic.pickups.to_dict(),
        "solves": optic.solves.to_dict(),
        "surface_group": optic.surface_group.to_dict(),
    }

    data["wavelengths"]["polarization"] = optic.polarization
    data["fields"]["field_type"] = optic.field_type
    data["fields"]["object_space_telecentric"] = optic.obj_space_telecentric
    return data


def optic_from_dict(data: dict) -> 'Optic':
    """Creates an Optic object from a dictionary representation.

    This function takes a dictionary (presumably created by `optic_to_dict`)
    and reconstructs an Optic object from it.

    Args:
        data: A dictionary containing the serialized representation of an
              Optic object.

    Returns:
        An Optic object reconstructed from the dictionary.
    """
    optic = Optic()  # Create a new Optic instance
    optic.aperture = Aperture.from_dict(data["aperture"]) if data.get("aperture") else None
    optic.surface_group = SurfaceGroup.from_dict(data["surface_group"])
    optic.fields = FieldGroup.from_dict(data["fields"])
    optic.wavelengths = WavelengthGroup.from_dict(data["wavelengths"])
    optic.pickups = PickupManager.from_dict(optic, data["pickups"])
    optic.solves = SolveManager.from_dict(optic, data["solves"])

    optic.polarization = data["wavelengths"].get("polarization", "ignore")
    optic.field_type = data["fields"].get("field_type")
    optic.obj_space_telecentric = data["fields"].get("object_space_telecentric", False)

    # Re-initialize dependent components
    optic.paraxial = Paraxial(optic)
    optic.aberrations = Aberrations(optic)
    # OpticUpdater is initialized within Optic.__init__ -> reset -> _initialize_attributes
    # RayGenerator was in the original from_dict, let's ensure it's initialized.
    # However, RayGenerator is typically initialized within RealRayTracer,
    # and RealRayTracer is initialized within Optic._initialize_attributes.
    # For now, let's stick to what was in the original from_dict.
    # If issues arise, this might need revisiting.
    optic.ray_generator = RayGenerator(optic)


    return optic
