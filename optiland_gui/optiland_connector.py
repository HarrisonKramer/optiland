import json
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QMessageBox # For showing errors to user

from optiland.optic import Optic 
from optiland.surfaces import Surface, ObjectSurface, ImageSurface
from optiland.physical_apertures.radial import RadialAperture, configure_aperture
from optiland.materials import IdealMaterial, Material as OptilandMaterial
import optiland.backend as be

class SpecialFloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if obj == float('inf'): return "Infinity"
            elif obj == float('-inf'): return "-Infinity"
            elif obj != obj: return "NaN"
        if hasattr(obj, 'item') and isinstance(obj.item(), float):
            val = obj.item()
            if val == float('inf'): return "Infinity"
            if val == float('-inf'): return "-Infinity"
            if val != val: return "NaN"
        if hasattr(be, 'isinf') and callable(be.isinf) and not isinstance(obj, (bool, int, float, str, list, dict, tuple, type(None))):
             try:
                if be.isinf(obj): # Check if be.isinf is the function
                    return "Infinity" if obj > 0 else "-Infinity"
             except: 
                pass
        try:
            return super().default(obj)
        except TypeError:
            if hasattr(obj, 'tolist'): return obj.tolist()
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

    DEFAULT_WAVELENGTH_UM = 0.550

    def __init__(self):
        super().__init__()
        self._optic = Optic("Default System")
        # For the very first launch, create the specific dummy system
        self._initialize_optic_structure(self._optic, is_specific_new_system=True)
        self._current_filepath = None
        self.opticLoaded.emit()

    def _initialize_optic_structure(self, optic_instance: Optic, is_specific_new_system: bool = False):
        """
        Initializes or ensures basic integrity of an Optic instance.
        If is_specific_new_system is True, it creates the defined dummy system.
        Otherwise, it ensures a loaded/existing system has minimal structure and valid wavelengths.
        """
        if is_specific_new_system:
            print("Connector: Creating specific new dummy system.")
            optic_instance.surface_group.surfaces.clear()
            optic_instance.wavelengths.wavelengths.clear() # Clear before adding new
            
            # Object Surface: Optiland's add_surface index 0 is object. Material is for space *after* it.
            # Thickness for object is distance to first optical surface.
            optic_instance.add_surface(surface_type='standard', radius=float('inf'), thickness=float('inf'), 
                                       comment="Object", index=0, material="Air")
            # Stop Surface (Standard, Plane)
            optic_instance.add_surface(surface_type='standard', radius=float('inf'), thickness=20.0, 
                                       comment="Stop", index=1, material="Air", is_stop=True)
            # Image Surface
            optic_instance.add_surface(surface_type='standard', radius=float('inf'), thickness=0.0, 
                                       comment="Image", index=2, material="Air")
            optic_instance.add_wavelength(0.550, is_primary=True, unit="um")
        else: # For loaded systems or general integrity checks
            if optic_instance.surface_group.num_surfaces < 2: # Must have at least Obj and Img
                print("Connector (Integrity): Optic has < 2 surfaces. Resetting to minimal Object/Image.")
                optic_instance.surface_group.surfaces.clear()
                optic_instance.wavelengths.wavelengths.clear() # Also clear WLs if resetting surfaces
                optic_instance.add_surface(surface_type='standard', radius=float('inf'), thickness=10.0, comment="Object", index=0, material="Air")
                optic_instance.add_surface(surface_type='standard', radius=float('inf'), thickness=0.0, comment="Image", index=1, material="Air")
                # Ensure wavelength for this minimal system
                if optic_instance.wavelengths.num_wavelengths == 0:
                    optic_instance.add_wavelength(self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um")

            # Wavelength integrity for already populated (but potentially flawed) optic
            if optic_instance.wavelengths.num_wavelengths == 0:
                print(f"Connector (Integrity): Optic has no wavelengths. Adding default primary: {self.DEFAULT_WAVELENGTH_UM} um.")
                optic_instance.add_wavelength(self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um")
            elif optic_instance.wavelengths.primary_index is None:
                print("Connector (Integrity): Optic has wavelengths but no primary. Setting first as primary.")
                if optic_instance.wavelengths.num_wavelengths > 0: # Should be true if primary_index is None but num_wavelengths isn't 0
                    optic_instance.wavelengths.wavelengths[0].is_primary = True
                    for i in range(1, optic_instance.wavelengths.num_wavelengths):
                        optic_instance.wavelengths.wavelengths[i].is_primary = False
        
        optic_instance.update() # Recalculate paraxial, etc.

    def _get_safe_primary_wavelength_value(self) -> float:
        if self._optic.wavelengths.num_wavelengths > 0:
            primary_idx = self._optic.wavelengths.primary_index
            if primary_idx is not None:
                try:
                    return self._optic.wavelengths.wavelengths[primary_idx].value
                except IndexError:
                    print("Warning: Primary wavelength index out of bounds. Attempting recovery.")
            # If no primary_idx or IndexError, try to fix and use the first one
            if self._optic.wavelengths.num_wavelengths > 0: # Check again in case list became empty
                print("Warning: Primary wavelength index issue or recovery needed. Using first wavelength.")
                self._optic.wavelengths.wavelengths[0].is_primary = True
                for i in range(1, self._optic.wavelengths.num_wavelengths):
                    self._optic.wavelengths.wavelengths[i].is_primary = False
                return self._optic.wavelengths.wavelengths[0].value
        
        print(f"Critical Warning: No valid wavelengths in optic. Falling back to default {self.DEFAULT_WAVELENGTH_UM} um.")
        # Ensure a default wavelength exists if we fall back here
        if self._optic.wavelengths.num_wavelengths == 0 :
            self._optic.add_wavelength(self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um")
            self._optic.update() # Update optic state
            return self.DEFAULT_WAVELENGTH_UM
        return self.DEFAULT_WAVELENGTH_UM


    def get_optic(self):
        return self._optic

    def new_system(self):
        self._optic = Optic("New Untitled System")
        # Call the initializer with the flag to create the specific dummy system
        self._initialize_optic_structure(self._optic, is_specific_new_system=True)
        self._current_filepath = None
        print("OpticConnector: New specific dummy system created.")
        self.opticLoaded.emit() # This will trigger LDE and viewer updates

    def load_optic_from_file(self, filepath):
        try:
            with open(filepath, 'r') as f:
                def json_inf_nan_hook(dct):
                    for k, v in dct.items():
                        if isinstance(v, str):
                            if v == "Infinity": dct[k] = float('inf')
                            elif v == "-Infinity": dct[k] = float('-inf')
                            elif v == "NaN": dct[k] = float('nan')
                    return dct

                data = json.load(f, object_hook=json_inf_nan_hook)

            self._optic = Optic.from_dict(data) 
            self._current_filepath = filepath
            print(f"OpticConnector: Optic loaded from {filepath}. Checking integrity.")
            
            # After loading, ensure basic integrity but do NOT force the dummy system structure
            self._initialize_optic_structure(self._optic, is_specific_new_system=False) 
            
            # Diagnostic check for stop surface after load and integrity check
            try:
                stop_idx_after_update = self._optic.surface_group.stop_index
                print(f"DEBUG: Stop index after load & integrity check is: {stop_idx_after_update}")
            except ValueError as e_stop:
                print(f"DEBUG ALERT: Error getting stop_index after load & integrity check: {e_stop}")
                QMessageBox.warning(None, "Load Warning", f"Loaded system from {filepath} may be missing a designated stop surface or has other structural issues:\n{e_stop}")


            self.opticLoaded.emit()

        except ValueError as ve: 
            error_message = f"Failed to process system from {filepath}:\n{ve}"
            print(f"OpticConnector: ValueError: {error_message}")
            QMessageBox.critical(None, "Load Error", error_message)
            self.new_system() # Revert to a new default system on critical error
        except Exception as e:
            error_message = f"An unexpected error occurred while loading {filepath}:\n{e}"
            print(f"OpticConnector: General error: {error_message}")
            QMessageBox.critical(None, "Load Error", error_message)
            self.new_system() # Revert to a new default system    
            
    def save_optic_to_file(self, filepath):
        try:
            # Ensure primary WL exists before saving
            if self._optic.wavelengths.num_wavelengths == 0:
                self._optic.add_wavelength(self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um")
            elif self._optic.wavelengths.primary_index is None and self._optic.wavelengths.num_wavelengths > 0 :
                 self._optic.wavelengths.wavelengths[0].is_primary = True


            data = self._optic.to_dict()
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4, cls=SpecialFloatEncoder)
            self._current_filepath = filepath
            print(f"OpticConnector: Optic saved to {filepath}")
        except Exception as e:
            print(f"OpticConnector: Error saving optic to {filepath}: {e}")
            QMessageBox.critical(None, "Save Error", f"Could not save system to {filepath}:\n{e}")

    def get_current_filepath(self):
        return self._current_filepath

    def get_surface_count(self):
        if not self._optic or not self._optic.surface_group:
            return 0
        return self._optic.surface_group.num_surfaces

    def get_column_headers(self):
        return [
            "Type",
            "Comment",
            "Radius",
            "Thickness",
            "Material",
            "Conic",
            "Semi-Diameter",
        ]

    def get_surface_data(self, row, col_idx):
        # This method should be the same as the one that fixed the TypeError
        if not (0 <= row < self.get_surface_count()):
            return None
        surface = self._optic.surface_group.surfaces[row]
        # ... (implementation from the version that fixed the TypeError)
        if col_idx == self.COL_TYPE:
            base_type = "Object" if row == 0 else \
                        "Image" if row == self.get_surface_count() - 1 else \
                        surface.surface_type if surface.surface_type else "Standard"
            is_intermediate_stop = surface.is_stop and not (row == 0 or row == self.get_surface_count() - 1)
            return f"Stop ({base_type})" if is_intermediate_stop else base_type
        elif col_idx == self.COL_COMMENT:
            return surface.comment
        elif col_idx == self.COL_RADIUS:
            radius = surface.geometry.radius
            val = float(radius.item()) if hasattr(radius, 'item') and not isinstance(radius, (float,int)) else float(radius)
            return "inf" if val == float('inf') else f"{val:.4f}"
        elif col_idx == self.COL_THICKNESS:
            if row < self.get_surface_count() - 1:
                thickness_val_arr = self._optic.surface_group.get_thickness(row)
                return f"{float(thickness_val_arr[0]):.4f}" if thickness_val_arr is not None and len(thickness_val_arr) > 0 else "N/A"
            return "N/A" 
        elif col_idx == self.COL_MATERIAL:
            if surface.is_reflective: return "Mirror"
            relevant_material = surface.material_post 
            if row == self.get_surface_count() - 1 and row > 0 : 
                relevant_material = self._optic.surface_group.surfaces[row-1].material_post
            if isinstance(relevant_material, IdealMaterial):
                wl_value = self._get_safe_primary_wavelength_value()
                n_val = relevant_material.n(wl_value)
                return "Air" if n_val == 1.0 else f"Ideal n={n_val:.4f}"
            if isinstance(relevant_material, OptilandMaterial):
                return relevant_material.name
            return "Unknown" 
        elif col_idx == self.COL_CONIC:
            k_val = surface.geometry.k if hasattr(surface.geometry, "k") else 0.0
            return f"{float(k_val):.4f}"
        elif col_idx == self.COL_SEMI_DIAMETER:
            ap = surface.aperture
            if isinstance(ap, RadialAperture):
                return f"{float(ap.r_max):.4f}"
            if surface.semi_aperture is not None:
                return f"{float(surface.semi_aperture):.4f}"
            return "Auto"
        return None


    def set_surface_data(self, row, col_idx, value_str):
        # This method should be the same as the one that fixed the TypeError
        if not (0 <= row < self.get_surface_count()):
            return
        try:
            surface = self._optic.surface_group.surfaces[row]
            updater = self._optic._updater 
            # ... (implementation from the version that fixed the TypeError)
            if col_idx == self.COL_COMMENT:
                surface.comment = value_str
            elif col_idx == self.COL_RADIUS:
                new_radius = (
                    float("inf") if value_str.lower() == "inf" else float(value_str)
                )
                updater.set_radius(new_radius, row)
            elif col_idx == self.COL_THICKNESS:
                if row < self.get_surface_count() - 1:
                    updater.set_thickness(float(value_str), row)
            elif col_idx == self.COL_MATERIAL:
                new_material_name = value_str.strip()
                if new_material_name.lower() == "mirror":
                    surface.is_reflective = True
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
                if row + 1 < self.get_surface_count():
                    self._optic.surface_group.surfaces[row + 1].material_pre = surface.material_post
            elif col_idx == self.COL_CONIC:
                if hasattr(surface.geometry, "k"):
                    updater.set_conic(float(value_str), row)
            elif col_idx == self.COL_SEMI_DIAMETER:
                try:
                    semi_diam = float(value_str)
                    surface.aperture = configure_aperture(semi_diam * 2.0) 
                except ValueError: 
                    surface.aperture = None 
            self._optic.update() 
            self.surfaceDataChanged.emit(row, col_idx, self.get_surface_data(row, col_idx))
            self.opticChanged.emit()
        except Exception as e:
            print(f"OpticConnector: Error setting data at ({row},{col_idx}) to '{value_str}': {e}")


    def add_surface(self, index=-1): 
        num_lde_rows = self.get_surface_count()
        optic_insert_idx = num_lde_rows - 1 
        if optic_insert_idx < 1 : 
             optic_insert_idx = 1 
        if index != -1 and 0 < index < num_lde_rows -1: 
             optic_insert_idx = index
        params = {
            "surface_type": "standard", "radius": float('inf'),
            "thickness": 5.0, "material": "Air", 
            "comment": "New Surface", "index": optic_insert_idx
        }
        self._optic.add_surface(**params)
        self._optic.update()
        self.surfaceAdded.emit(optic_insert_idx) 
        self.surfaceCountChanged.emit()
        self.opticChanged.emit()

    def remove_surface(self, lde_row_index):
        optic_surface_index = lde_row_index 
        if 0 < optic_surface_index < self.get_surface_count() - 1:
            self._optic.surface_group.remove_surface(optic_surface_index)
            self._optic.update()
            self.surfaceRemoved.emit(lde_row_index)
            self.surfaceCountChanged.emit()
            self.opticChanged.emit()
        else:
            print("OpticConnector: Cannot remove Object or Image surface via this LDE action.")
