from optiland.aperture import Aperture
from optiland.fields import FieldGroup, FieldType
from optiland.pickup import PickupManager
from optiland.rays import PolarizationState, PolarizationType # Added
from optiland.solves import SolveManager
from optiland.surfaces import SurfaceGroup
from optiland.wavelength import WavelengthGroup

class OpticSerializer:
    def __init__(self, optic):
        self.optic = optic

    def to_dict(self):
        """Convert the optical system to a dictionary.

        Returns:
            dict: The dictionary representation of the optical system.
        """
        data = {
            "version": 1.0,
            "aperture": self.optic.aperture.to_dict() if hasattr(self.optic, 'aperture') and self.optic.aperture else None,
            "fields": self.optic.fields.to_dict() if hasattr(self.optic, 'fields') and self.optic.fields else {},
            "wavelengths": self.optic.wavelengths.to_dict() if hasattr(self.optic, 'wavelengths') and self.optic.wavelengths else {},
            "pickups": self.optic.pickups.to_dict() if hasattr(self.optic, 'pickups') and self.optic.pickups else {},
            "solves": self.optic.solves.to_dict() if hasattr(self.optic, 'solves') and self.optic.solves else {},
            "surface_group": self.optic.surface_group.to_dict() if hasattr(self.optic, 'surface_group') and self.optic.surface_group else {},
        }

        # Add direct attributes of Optic that were previously nested in the dicts by Optic.to_dict
        if hasattr(self.optic, 'polarization'):
            if isinstance(self.optic.polarization, PolarizationType):
                data["wavelengths"]["polarization"] = self.optic.polarization.value
            elif isinstance(self.optic.polarization, PolarizationState):
                data["wavelengths"]["polarization"] = self.optic.polarization.to_dict()
            else: # Should not happen if Optic.polarization is correctly typed and set
                data["wavelengths"]["polarization"] = str(self.optic.polarization)

        if hasattr(self.optic, 'field_type') and self.optic.field_type is not None:
            data["fields"]["field_type"] = self.optic.field_type.value
        elif hasattr(self.optic, 'field_type') and self.optic.field_type is None:
             data["fields"]["field_type"] = None # Explicitly store None

        if hasattr(self.optic, 'obj_space_telecentric'):
            data["fields"]["object_space_telecentric"] = self.optic.obj_space_telecentric

        return data

    @staticmethod
    def from_dict(data, cls):
        """Create an optical system from a dictionary.

        Args:
            data (dict): The dictionary representation of the optical system.
            cls (type): The class to instantiate (e.g., Optic).

        Returns:
            An instance of cls (e.g., Optic).
        """
        optic = cls()  # Optic's __init__ calls reset, which calls _initialize_attributes

        # Deserialize components using their respective from_dict @classmethods
        if "aperture" in data and data["aperture"] is not None:
            optic.aperture = Aperture.from_dict(data["aperture"])

        if "fields" in data:
            optic.fields = FieldGroup.from_dict(data["fields"])
            if "field_type" in data["fields"] and data["fields"]["field_type"] is not None:
                try:
                    optic.field_type = FieldType(data["fields"]["field_type"])
                except ValueError: # Handle old data or invalid strings
                    optic.field_type = None # Or a default, or raise error
            else:
                optic.field_type = None # if key exists but is None

            if "object_space_telecentric" in data["fields"]:
                optic.obj_space_telecentric = data["fields"]["object_space_telecentric"]

        if "wavelengths" in data:
            optic.wavelengths = WavelengthGroup.from_dict(data["wavelengths"])
            if "polarization" in data["wavelengths"]:
                polarization_data = data["wavelengths"]["polarization"]
                if isinstance(polarization_data, str):
                    try:
                        optic.polarization = PolarizationType(polarization_data)
                    except ValueError: # If string is not a valid PolarizationType value (e.g. old format)
                        # This case might need more specific handling if there are other legacy string values
                        # For now, default to IGNORE if unknown string
                        optic.polarization = PolarizationType.IGNORE
                elif isinstance(polarization_data, dict): # Assumed to be a PolarizationState dict
                    optic.polarization = PolarizationState.from_dict(polarization_data)
                else: # Default or error for other unexpected data types
                    optic.polarization = PolarizationType.IGNORE
            else: # If "polarization" key is missing
                optic.polarization = PolarizationType.IGNORE


        if "surface_group" in data:
            optic.surface_group = SurfaceGroup.from_dict(data["surface_group"])

        # Optic._initialize_attributes already creates self.pickups and self.solves instances.
        # Their from_dict methods should populate these existing manager instances.
        # PickupManager and SolveManager from_dict are also @classmethods, so this needs care.
        # The original Optic.from_dict was:
        # optic.pickups = PickupManager.from_dict(optic, data["pickups"])
        # optic.solves = SolveManager.from_dict(optic, data["solves"])
        # This implies their from_dict takes `optic` and `data`, and returns a new manager instance.
        # This contradicts Optic._initialize_attributes creating them.
        # For now, I will follow the original Optic.from_dict structure for pickups and solves.
        if "pickups" in data:
            optic.pickups = PickupManager.from_dict(optic, data["pickups"])

        if "solves" in data:
            optic.solves = SolveManager.from_dict(optic, data["solves"])

        # Paraxial, Aberrations, RealRayTracer (which includes RayGenerator)
        # are initialized within Optic._initialize_attributes when cls() is called.
        # Their state depends on the other optic components that have just been set.

        return optic
