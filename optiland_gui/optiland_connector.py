# optiland_gui/optiland_connector.py
import json
from PySide6.QtCore import QObject, Signal, Slot

from optiland.optic import Optic # Actual Optiland import
from optiland.surfaces import Surface, ObjectSurface # For type checking
from optiland.physical_apertures.radial import RadialAperture, configure_aperture
from optiland.materials import IdealMaterial, Material as OptilandMaterial 
import optiland.backend as be # For be.inf if Optiland uses it

# Helper for JSON serialization of float('inf') and other special float values
class SpecialFloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if obj == float('inf'):
                return "Infinity"
            elif obj == float('-inf'):
                return "-Infinity"
            elif obj != obj:  # NaN
                return "NaN"
        # Handle numpy float types if Optiland backend uses them and they reach here
        if hasattr(obj, 'item') and isinstance(obj.item(), float):
            val = obj.item()
            if val == float('inf'): return "Infinity"
            if val == float('-inf'): return "-Infinity"
            if val != val: return "NaN"
        # Handle cases where be.inf might be a specific object instance
        if be.isinf(obj) and obj > 0 : return "Infinity" # Check if be.isinf is the function
        if be.isinf(obj) and obj < 0 : return "-Infinity"


        try:
            return super().default(obj)
        except TypeError:
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return str(obj)

class OptilandConnector(QObject):
    opticLoaded = Signal()
    opticChanged = Signal()
    surfaceDataChanged = Signal(int, int, object)
    surfaceAdded = Signal(int)
    surfaceRemoved = Signal(int)
    surfaceCountChanged = Signal()

    COL_TYPE = 0
    COL_COMMENT = 1
    COL_RADIUS = 2
    COL_THICKNESS = 3
    COL_MATERIAL = 4
    COL_CONIC = 5
    COL_SEMI_DIAMETER = 6

    DEFAULT_WAVELENGTH_UM = 0.550 # e.g., 550 nm

    def __init__(self):
        super().__init__()
        self._optic = Optic("Default System")
        self._ensure_basic_optic_structure(self._optic)
        self._current_filepath = None
        # Emit opticLoaded AFTER the basic structure is ensured and optic is ready.
        self.opticLoaded.emit()

    def _ensure_basic_optic_structure(self, optic_instance: Optic):
        if optic_instance.surface_group.num_surfaces == 0:
            print("Connector: Optic has no surfaces. Adding default object/image.")
            # Optic.add_surface uses index to insert, 0 is object
            optic_instance.add_surface(surface_type='standard', radius=float('inf'), thickness=10.0, comment="Object", index=0, material="Air")
            optic_instance.add_surface(surface_type='standard', radius=float('inf'), thickness=0.0, comment="Image", index=1, material="Air")

        if optic_instance.wavelengths.num_wavelengths == 0:
            print(f"Connector: Optic has no wavelengths. Adding default primary: {self.DEFAULT_WAVELENGTH_UM} um.")
            optic_instance.add_wavelength(self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um")
        elif optic_instance.wavelengths.primary_index is None:
            print("Connector: Optic has wavelengths but no primary. Setting first as primary.")
            if optic_instance.wavelengths.num_wavelengths > 0:
                optic_instance.wavelengths.wavelengths[0].is_primary = True
                for i in range(1, optic_instance.wavelengths.num_wavelengths):
                    optic_instance.wavelengths.wavelengths[i].is_primary = False
        
        optic_instance.update() # Recalculate paraxial, etc.

    def _get_safe_primary_wavelength_value(self) -> float:
        """Safely retrieves the primary wavelength value, defaulting if necessary."""
        if self._optic.wavelengths.num_wavelengths > 0:
            primary_idx = self._optic.wavelengths.primary_index
            if primary_idx is not None:
                try:
                    return self._optic.wavelengths.wavelengths[primary_idx].value
                except IndexError: # Should not happen if primary_idx is not None and num_wavelengths > 0
                    pass 
            # If no primary_idx or the above failed, try to set and use the first one
            print("Warning: Primary wavelength index issue. Attempting to use first wavelength.")
            self._optic.wavelengths.wavelengths[0].is_primary = True
            for i in range(1, self._optic.wavelengths.num_wavelengths): # Ensure only one primary
                self._optic.wavelengths.wavelengths[i].is_primary = False
            return self._optic.wavelengths.wavelengths[0].value
        
        # This case means the wavelength list is empty, which _ensure_basic_optic_structure should prevent.
        print(f"Critical Warning: No wavelengths in optic. Falling back to default {self.DEFAULT_WAVELENGTH_UM} um.")
        return self.DEFAULT_WAVELENGTH_UM

    def get_optic(self):
        return self._optic

    def new_system(self):
        self._optic = Optic("Untitled System")
        self._ensure_basic_optic_structure(self._optic)
        self._current_filepath = None
        print("OpticConnector: New system created.")
        self.opticLoaded.emit()
        self.opticChanged.emit()

    def load_optic_from_file(self, filepath):
        try:
            with open(filepath, 'r') as f:
                # Handle "Infinity" and "NaN" during loading
                def json_inf_nan_hook(dct):
                    for k, v in dct.items():
                        if isinstance(v, str):
                            if v == "Infinity":
                                dct[k] = float('inf')
                            elif v == "-Infinity":
                                dct[k] = float('-inf')
                            elif v == "NaN":
                                dct[k] = float('nan')
                    return dct
                data = json.load(f, object_hook=json_inf_nan_hook)

            self._optic = Optic.from_dict(data)
            self._ensure_basic_optic_structure(self._optic) # Ensure basics even after load
            self._current_filepath = filepath
            print(f"OpticConnector: Optic loaded from {filepath}")
            self.opticLoaded.emit()
            self.opticChanged.emit()
        except Exception as e:
            print(f"OpticConnector: Error loading optic from {filepath}: {e}")

    def save_optic_to_file(self, filepath):
        try:
            self._ensure_basic_optic_structure(self._optic) # Ensure primary WL exists before saving
            data = self._optic.to_dict()
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4, cls=SpecialFloatEncoder)
            self._current_filepath = filepath
            print(f"OpticConnector: Optic saved to {filepath}")
        except Exception as e:
            print(f"OpticConnector: Error saving optic to {filepath}: {e}")
    
    def get_current_filepath(self):
        return self._current_filepath

    def get_surface_count(self):
        if not self._optic or not self._optic.surface_group: return 0
        return self._optic.surface_group.num_surfaces

    def get_column_headers(self):
        return ["Type", "Comment", "Radius", "Thickness", "Material", "Conic", "Semi-Diameter"]

    def get_surface_data(self, row, col_idx):
        if not (0 <= row < self.get_surface_count()):
            return None
        
        surface = self._optic.surface_group.surfaces[row]

        if col_idx == self.COL_TYPE:
            base_type = "Object" if row == 0 else \
                        "Image" if row == self.get_surface_count() - 1 else \
                        surface.surface_type if surface.surface_type else "Standard"
            # Display STOP only if it's not also Object or Image
            is_intermediate_stop = surface.is_stop and not (row == 0 or row == self.get_surface_count() - 1)
            return f"Stop ({base_type})" if is_intermediate_stop else base_type
        elif col_idx == self.COL_COMMENT:
            return surface.comment
        elif col_idx == self.COL_RADIUS:
            radius = surface.geometry.radius
            val = float(radius.item()) if hasattr(radius, 'item') else float(radius)
            return "inf" if val == float('inf') else f"{val:.4f}"
        elif col_idx == self.COL_THICKNESS:
            if row < self.get_surface_count() - 1:
                thickness_val_arr = self._optic.surface_group.get_thickness(row)
                # Ensure thickness_val_arr is not None and has elements
                return f"{float(thickness_val_arr[0]):.4f}" if thickness_val_arr is not None and len(thickness_val_arr) > 0 else "N/A"
            return "N/A" # No thickness after the last (image) surface
        elif col_idx == self.COL_MATERIAL:
            if surface.is_reflective: return "Mirror"
            
            relevant_material = surface.material_post # Material of space *after* this surface
            if row == self.get_surface_count() - 1 and row > 0: # Image surface (not object also)
                relevant_material = self._optic.surface_group.surfaces[row - 1].material_post # Material before image

            if isinstance(relevant_material, IdealMaterial):
                wl_value = self._get_safe_primary_wavelength_value()
                n_val = relevant_material.n(wl_value)
                return "Air" if n_val == 1.0 else f"Ideal n={n_val:.4f}"
            if isinstance(relevant_material, OptilandMaterial):
                return relevant_material.name
            return "Unknown"
        elif col_idx == self.COL_CONIC:
            k_val = surface.geometry.k if hasattr(surface.geometry, 'k') else 0.0
            return f"{float(k_val):.4f}"
        elif col_idx == self.COL_SEMI_DIAMETER:
            ap = surface.aperture
            if isinstance(ap, RadialAperture):
                return f"{float(ap.r_max):.4f}"
            # Optiland's OpticUpdater.update_paraxial() sets surface.semi_aperture
            # This is usually called by optic.update()
            # self._optic.update_paraxial() # Might be too heavy here, ensure it's called after changes
            if surface.semi_aperture is not None:
                 return f"{float(surface.semi_aperture):.4f}"
            return "Auto"
        return None

    def set_surface_data(self, row, col_idx, value_str):
        if not (0 <= row < self.get_surface_count()):
            return

        try:
            surface = self._optic.surface_group.surfaces[row]
            updater = self._optic._updater 

            if col_idx == self.COL_COMMENT:
                surface.comment = value_str
            elif col_idx == self.COL_RADIUS:
                new_radius = float('inf') if value_str.lower() == 'inf' else float(value_str)
                updater.set_radius(new_radius, row)
            elif col_idx == self.COL_THICKNESS:
                if row < self.get_surface_count() - 1:
                    updater.set_thickness(float(value_str), row)
            elif col_idx == self.COL_MATERIAL:
                new_material_name = value_str.strip()
                
                if new_material_name.lower() == "mirror":
                    surface.is_reflective = True
                    # Material post should be same as pre for mirrors to avoid issues if refractive index is used
                    surface.material_post = surface.material_pre if surface.material_pre else IdealMaterial(n=1.0)
                else:
                    surface.is_reflective = False
                    if new_material_name.lower() == "air":
                        surface.material_post = IdealMaterial(n=1.0)
                    else:
                        try:
                            n_val = float(new_material_name)
                            surface.material_post = IdealMaterial(n=n_val)
                        except ValueError:
                            surface.material_post = OptilandMaterial(name=new_material_name)
                
                # Update the material_pre of the *next* surface
                if row + 1 < self.get_surface_count():
                    self._optic.surface_group.surfaces[row + 1].material_pre = surface.material_post

            elif col_idx == self.COL_CONIC:
                if hasattr(surface.geometry, 'k'):
                     updater.set_conic(float(value_str), row)
            elif col_idx == self.COL_SEMI_DIAMETER:
                try:
                    semi_diam = float(value_str)
                    surface.aperture = configure_aperture(semi_diam * 2.0) 
                except ValueError: # Handle "Auto" or invalid input
                    surface.aperture = None 
                    print(f"Semi-diameter for surface {row} set to auto (aperture removed).")
            
            self._optic.update() # Crucial: apply solves, pickups, update paraxial data
            self.surfaceDataChanged.emit(row, col_idx, self.get_surface_data(row, col_idx))
            self.opticChanged.emit()

        except Exception as e:
            print(f"OpticConnector: Error setting data at ({row},{col_idx}) to '{value_str}': {e}")
            # Consider emitting an error signal to the GUI

    def add_surface(self, index=-1): # index is LDE index
        num_lde_rows = self.get_surface_count()
        
        # Determine Optic surface_group index for insertion
        # Add before Image plane, which is the last row in LDE
        optic_insert_idx = num_lde_rows - 1 
        if optic_insert_idx < 1 : # Cannot insert before first optical surface (idx 1)
             optic_insert_idx = 1 
        if index != -1 and 0 < index < num_lde_rows -1: # if specific intermediate row is given
             optic_insert_idx = index


        params = {
            "surface_type": "standard", "radius": float('inf'),
            "thickness": 5.0, "material": "Air", # material is for the space *after* new surface
            "comment": "New Surface", "index": optic_insert_idx
        }
        self._optic.add_surface(**params)
        self._optic.update()
        
        # The LDE index where it appears is optic_insert_idx
        self.surfaceAdded.emit(optic_insert_idx) 
        self.surfaceCountChanged.emit()
        self.opticChanged.emit()

    def remove_surface(self, lde_row_index):
        optic_surface_index = lde_row_index 
        # Prevent removing Object (LDE row 0 / Optic idx 0) 
        # or Image (LDE row N-1 / Optic idx N-1)
        if 0 < optic_surface_index < self.get_surface_count() - 1:
            self._optic.surface_group.remove_surface(optic_surface_index)
            self._optic.update()
            self.surfaceRemoved.emit(lde_row_index) 
            self.surfaceCountChanged.emit()
            self.opticChanged.emit()
        else:
            print("OpticConnector: Cannot remove Object or Image surface via this LDE action.")