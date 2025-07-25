"""Defines the connector that bridges the GUI and the Optiland core logic.

This module contains the `OptilandConnector` class, which acts as a vital
intermediary between the user interface and the underlying `optiland` optical
engine. It manages the active optical system (`Optic` object), handles file
I/O, provides data to UI components like the Lens Editor, and manages the
undo/redo stack.

Author: Manuel Fragata Mendes, 2025
"""

import json

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox

from optiland.materials import IdealMaterial
from optiland.materials import Material as OptilandMaterial
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.physical_apertures.radial import configure_aperture
from optiland_gui.undo_redo_manager import UndoRedoManager


class SpecialFloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if obj == float("inf"):
                return "Infinity"
            elif obj == float("-inf"):
                return "-Infinity"
            elif obj != obj:
                return "NaN"
        if hasattr(obj, "item") and isinstance(obj.item(), float):
            val = obj.item()
            if val == float("inf"):
                return "Infinity"
            if val == float("-inf"):
                return "-Infinity"
            if val != val:
                return "NaN"
        try:
            return super().default(obj)
        except TypeError:
            if hasattr(obj, "tolist"):
                return obj.tolist()
            return str(obj)


class OptilandConnector(QObject):
    """The main bridge between the Optiland core logic and the GUI.

    This class manages the `Optic` object, which holds all the optical system
    data. It exposes methods for creating, loading, and saving systems, and for
    querying and modifying surface data. It uses Qt signals to notify the rest
    of the GUI about changes, ensuring that all panels stay synchronized. It also
    manages the undo/redo functionality.

    An instance of this class is often made available to scripting environments
    (e.g., a Python console) as 'connector'.

    Signals:
        opticLoaded: Emitted when a new optical system is created or loaded.
        opticChanged: Emitted whenever any part of the optical system changes.
        modifiedStateChanged: Emitted when the system's modified status changes.
        surfaceDataChanged: Emitted when a specific cell in the lens data changes.
        surfaceCountChanged: Emitted when a surface is added or removed.
        undoStackAvailabilityChanged: Emitted when the undo stack becomes
                                        available/unavailable.
        redoStackAvailabilityChanged: Emitted when the redo stack becomes
                                        available/unavailable.
    """

    opticLoaded = Signal()
    opticChanged = Signal()
    modifiedStateChanged = Signal(bool)
    surfaceDataChanged = Signal(int, int, object)
    surfaceAdded = Signal(int)
    surfaceRemoved = Signal(int)
    surfaceCountChanged = Signal()
    undoStackAvailabilityChanged = Signal(bool)
    redoStackAvailabilityChanged = Signal(bool)

    COL_TYPE = 0
    COL_COMMENT = 1
    COL_RADIUS = 2
    COL_THICKNESS = 3
    COL_MATERIAL = 4
    COL_CONIC = 5
    COL_SEMI_DIAMETER = 6

    DEFAULT_WAVELENGTH_UM = 0.550

    def __init__(self):
        """Initializes the OptilandConnector.

        Sets up a default, empty optical system, initializes the undo/redo manager,
        and connects signals for stack availability.
        """
        super().__init__()
        self._optic = Optic("Default System")
        self._undo_redo_manager = UndoRedoManager(self)
        self._initialize_optic_structure(self._optic, is_specific_new_system=True)
        self._current_filepath = None
        self._is_modified = False

        self._undo_redo_manager.undoStackAvailabilityChanged.connect(
            self.undoStackAvailabilityChanged
        )
        self._undo_redo_manager.redoStackAvailabilityChanged.connect(
            self.redoStackAvailabilityChanged
        )
        self.opticLoaded.emit()
        self._undo_redo_manager.clear_stacks()

    def set_modified(self, modified: bool):
        """Sets the modified state of the system and emits a signal if it changes.

        Args:
            modified: True if the system has unsaved changes, False otherwise.
        """
        if self._is_modified != modified:
            self._is_modified = modified
            self.modifiedStateChanged.emit(self._is_modified)

    def is_modified(self) -> bool:
        """Checks if the current optical system has unsaved changes.

        Returns:
            True if the optic has been modified since the last save, False otherwise.
        """
        return self._is_modified

    def _initialize_optic_structure(
        self, optic_instance: Optic, is_specific_new_system: bool = False
    ):
        """Ensures the optic has a valid basic structure.

        This method checks for a minimum number of surfaces (Object, Image) and
        at least one primary wavelength. If the structure is invalid, it adds
        the necessary components to ensure the system is in a usable state.

        Args:
            optic_instance: The `Optic` object to validate and initialize.
            is_specific_new_system: If True, creates a default doublet-like
                                    structure. Otherwise, it just ensures integrity.
        """
        if is_specific_new_system:
            optic_instance.surface_group.surfaces.clear()
            optic_instance.wavelengths.wavelengths.clear()

            optic_instance.add_surface(
                surface_type="standard",
                radius=float("inf"),
                thickness=float("inf"),
                comment="Object",
                index=0,
                material="Air",
            )
            optic_instance.add_surface(
                surface_type="standard",
                radius=float("inf"),
                thickness=20.0,
                comment="Stop",
                index=1,
                material="Air",
                is_stop=True,
            )
            optic_instance.add_surface(
                surface_type="standard",
                radius=float("inf"),
                thickness=0.0,
                comment="Image",
                index=2,
                material="Air",
            )
            optic_instance.add_wavelength(0.550, is_primary=True, unit="um")
        else:
            if optic_instance.surface_group.num_surfaces < 2:
                print(
                    "Connector (Integrity): Optic has < 2 surfaces. "
                    "Resetting to minimal Object/Image."
                )
                optic_instance.surface_group.surfaces.clear()
                optic_instance.wavelengths.wavelengths.clear()
                optic_instance.add_surface(
                    surface_type="standard",
                    radius=float("inf"),
                    thickness=10.0,
                    comment="Object",
                    index=0,
                    material="Air",
                )
                optic_instance.add_surface(
                    surface_type="standard",
                    radius=float("inf"),
                    thickness=0.0,
                    comment="Image",
                    index=1,
                    material="Air",
                )
                if optic_instance.wavelengths.num_wavelengths == 0:
                    optic_instance.add_wavelength(
                        self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um"
                    )

            if optic_instance.wavelengths.num_wavelengths == 0:
                print(
                    f"Connector (Integrity): Optic has no wavelengths. "
                    f"Adding default primary: {self.DEFAULT_WAVELENGTH_UM} um."
                )
                optic_instance.add_wavelength(
                    self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um"
                )
            elif optic_instance.wavelengths.primary_index is None:
                print(
                    "Connector (Integrity): Optic has wavelengths but no primary. "
                    "Setting first as primary."
                )
                if optic_instance.wavelengths.num_wavelengths > 0:
                    optic_instance.wavelengths.wavelengths[0].is_primary = True
                    for i in range(1, optic_instance.wavelengths.num_wavelengths):
                        optic_instance.wavelengths.wavelengths[i].is_primary = False

        optic_instance.update()

    def _get_safe_primary_wavelength_value(self) -> float:
        """Gets the primary wavelength value, with fallbacks for safety.

        This method attempts to retrieve the primary wavelength. If none is set or
        an inconsistency is found, it will attempt to recover by setting the first
        wavelength as primary or by adding a default wavelength if none exist.

        Returns:
            The value of the primary wavelength in micrometers.
        """
        if self._optic.wavelengths.num_wavelengths > 0:
            primary_idx = self._optic.wavelengths.primary_index
            if primary_idx is not None:
                try:
                    return self._optic.wavelengths.wavelengths[primary_idx].value
                except IndexError:
                    print(
                        "Warning: Primary wavelength index out of bounds. "
                        "Attempting recovery."
                    )
            if self._optic.wavelengths.num_wavelengths > 0:
                print(
                    "Warning: Primary wavelength index issue or recovery needed. "
                    "Using first wavelength."
                )
                self._optic.wavelengths.wavelengths[0].is_primary = True
                for i in range(1, self._optic.wavelengths.num_wavelengths):
                    self._optic.wavelengths.wavelengths[i].is_primary = False
                return self._optic.wavelengths.wavelengths[0].value

        print(
            f"Critical Warning: No valid wavelengths in optic. "
            f"Falling back to default {self.DEFAULT_WAVELENGTH_UM} um."
        )
        if self._optic.wavelengths.num_wavelengths == 0:
            self._optic.add_wavelength(
                self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um"
            )
            self._optic.update()
            return self.DEFAULT_WAVELENGTH_UM
        return self.DEFAULT_WAVELENGTH_UM

    def get_optic(self) -> Optic:
        """Returns the currently active Optic object.

        This is the main method for other parts of the application to get access
        to the optical system's data and methods.

        Returns:
            The active `Optic` instance.
        """
        return self._optic

    def _capture_optic_state(self):
        """Captures the current state of the optic as a dictionary.

        This is used by the undo/redo manager to save a snapshot of the system
        before a change is made. It ensures the optic has a valid primary
        wavelength before serialization.

        Returns:
            A dictionary representing the complete state of the `Optic` object.
        """
        if self._optic.wavelengths.num_wavelengths == 0:
            self._optic.add_wavelength(
                self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um"
            )
            self._optic.update()
        elif (
            self._optic.wavelengths.primary_index is None
            and self._optic.wavelengths.num_wavelengths > 0
        ):
            self._optic.wavelengths.wavelengths[0].is_primary = True
            self._optic.update()
        return self._optic.to_dict()

    def _restore_optic_state(self, state_data):
        """Restores the optic's state from a dictionary.

        This is used by the undo/redo manager to revert the system to a previous
        state. It reconstructs the `Optic` object from the dictionary and emits
        a signal to notify the GUI of the change.

        Args:
            state_data: A dictionary representing the state to restore.
        """
        self._optic = Optic.from_dict(state_data)
        self._initialize_optic_structure(self._optic, is_specific_new_system=False)
        self.opticLoaded.emit()

    def new_system(self):
        """Creates a new, default optical system.

        This method clears the current system and its undo/redo history, then
        initializes a fresh system with a default structure.
        """
        self._undo_redo_manager.clear_stacks()
        self._optic = Optic("New Untitled System")
        self._initialize_optic_structure(self._optic, is_specific_new_system=True)
        self._current_filepath = None
        print("OpticConnector: New specific dummy system created.")
        self.set_modified(False)
        self.opticLoaded.emit()

    def load_optic_from_file(self, filepath: str):
        """Loads an optical system from a JSON file.

        This method reads a JSON file, reconstructs an `Optic` object from it,
        and sets it as the current system. It performs integrity checks after
        loading and handles potential errors during the process.

        Args:
            filepath: The path to the JSON file to load.
        """
        try:
            with open(filepath) as f:

                def json_inf_nan_hook(dct):
                    for k, v in dct.items():
                        if isinstance(v, str):
                            if v == "Infinity":
                                dct[k] = float("inf")
                            elif v == "-Infinity":
                                dct[k] = float("-inf")
                            elif v == "NaN":
                                dct[k] = float("nan")
                    return dct

                data = json.load(f, object_hook=json_inf_nan_hook)

            self._undo_redo_manager.clear_stacks()
            self._optic = Optic.from_dict(data)
            self._current_filepath = filepath
            print(f"OpticConnector: Optic loaded from {filepath}. Checking integrity.")

            self._initialize_optic_structure(self._optic, is_specific_new_system=False)

            try:
                pass
            except ValueError as e_stop:
                print(
                    f"DEBUG ALERT: Error getting stop_index after "
                    f"load & integrity check: {e_stop}"
                )
                QMessageBox.warning(
                    None,
                    "Load Warning",
                    f"Loaded system from {filepath} may be missing a designated "
                    f"stop surface or has other structural issues:\n{e_stop}",
                )
            self.set_modified(False)
            self.opticLoaded.emit()

        except (ValueError, json.JSONDecodeError) as e:
            error_message = f"Failed to process system from {filepath}:\n{e}"
            print(f"OpticConnector: Error: {error_message}")
            QMessageBox.critical(None, "Load Error", error_message)
            self.new_system()
        except Exception as e:
            error_message = (
                f"An unexpected error occurred while loading {filepath}:\n{e}"
            )
            print(f"OpticConnector: General error: {error_message}")
            QMessageBox.critical(None, "Load Error", error_message)
            self.new_system()

    def save_optic_to_file(self, filepath: str):
        """Saves the current optical system to a JSON file.

        This method serializes the current `Optic` object to a dictionary and
        writes it to a JSON file. It handles special float values like infinity
        and NaN.

        Args:
            filepath: The path to the file where the system will be saved.
        """
        try:
            if self._optic.wavelengths.num_wavelengths == 0:
                self._optic.add_wavelength(
                    self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um"
                )
            elif (
                self._optic.wavelengths.primary_index is None
                and self._optic.wavelengths.num_wavelengths > 0
            ):
                self._optic.wavelengths.wavelengths[0].is_primary = True

            data = self._optic.to_dict()
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4, cls=SpecialFloatEncoder)
            self._current_filepath = filepath
            self.set_modified(False)
            print(f"OpticConnector: Optic saved to {filepath}")
        except Exception as e:
            print(f"OpticConnector: Error saving optic to {filepath}: {e}")
            QMessageBox.critical(
                None, "Save Error", f"Could not save system to {filepath}:\n{e}"
            )

    def get_current_filepath(self) -> str | None:
        """Returns the file path of the currently loaded system.

        Returns:
            The path as a string, or None if the system hasn't been saved or loaded.
        """
        return self._current_filepath

    def get_surface_count(self) -> int:
        """Returns the total number of surfaces in the system.

        Returns:
            The number of surfaces.
        """
        if not self._optic or not self._optic.surface_group:
            return 0
        return self._optic.surface_group.num_surfaces

    def get_column_headers(self) -> list[str]:
        """Returns the column headers for the Lens Data Editor table.

        Returns:
            A list of strings representing the column headers.
        """
        return [
            "Type",
            "Comment",
            "Radius",
            "Thickness",
            "Material",
            "Conic",
            "Semi-Diameter",
        ]

    def get_surface_data(self, row: int, col_idx: int):
        """Gets a specific data point from a surface for display in the GUI.

        This method retrieves a property from a surface and formats it as a
        string suitable for display in the Lens Editor table.

        Args:
            row: The index of the surface.
            col_idx: The index of the column (property) to retrieve.

        Returns:
            The formatted data as a string, or None if the indices are invalid.
        """
        if not (0 <= row < self.get_surface_count()):
            return None
        surface = self._optic.surface_group.surfaces[row]

        if col_idx == self.COL_TYPE:
            base_type = (
                "Object"
                if row == 0
                else (
                    "Image"
                    if row == self.get_surface_count() - 1
                    else surface.surface_type
                    if surface.surface_type
                    else "Standard"
                )
            )
            is_intermediate_stop = surface.is_stop and not (
                row == 0 or row == self.get_surface_count() - 1
            )
            return f"Stop ({base_type})" if is_intermediate_stop else base_type
        elif col_idx == self.COL_COMMENT:
            return surface.comment
        elif col_idx == self.COL_RADIUS:
            radius = surface.geometry.radius
            val = (
                float(radius.item())
                if hasattr(radius, "item") and not isinstance(radius, (float, int))
                else float(radius)
            )
            return "inf" if val == float("inf") else f"{val:.4f}"
        elif col_idx == self.COL_THICKNESS:
            if row < self.get_surface_count() - 1:
                thickness_val_arr = self._optic.surface_group.get_thickness(row)
                return (
                    f"{float(thickness_val_arr[0]):.4f}"
                    if thickness_val_arr is not None and len(thickness_val_arr) > 0
                    else "N/A"
                )
            return "N/A"
        elif col_idx == self.COL_MATERIAL:
            if surface.is_reflective:
                return "Mirror"
            relevant_material = surface.material_post
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

    def set_surface_data(self, row: int, col_idx: int, value_str: str):
        """Sets a specific property of a surface based on user input from the GUI.

        This method takes a string value from the Lens Editor, parses it, and
        updates the corresponding property in the `Optic` object. It also saves
        the state for undo/redo and emits signals to update the GUI.

        Args:
            row: The index of the surface to modify.
            col_idx: The index of the column (property) to modify.
            value_str: The new value as a string from the UI.
        """
        if not (0 <= row < self.get_surface_count()):
            return
        try:
            old_state = self._capture_optic_state()
            surface = self._optic.surface_group.surfaces[row]
            updater = self._optic._updater

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
                    surface.material_post = (
                        surface.material_pre
                        if surface.material_pre
                        else IdealMaterial(n=1.0)
                    )
                else:
                    surface.is_reflective = False
                    if new_material_name.lower() == "air":
                        surface.material_post = IdealMaterial(n=1.0)
                    else:
                        try:
                            n_val = float(new_material_name)
                            surface.material_post = IdealMaterial(n=n_val)
                        except ValueError:
                            surface.material_post = OptilandMaterial(
                                name=new_material_name
                            )
                if row + 1 < self.get_surface_count():
                    self._optic.surface_group.surfaces[
                        row + 1
                    ].material_pre = surface.material_post
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
            self._undo_redo_manager.add_state(old_state)
            self.set_modified(True)
            self.surfaceDataChanged.emit(
                row, col_idx, self.get_surface_data(row, col_idx)
            )
            self.opticChanged.emit()
        except Exception as e:
            print(
                f"OpticConnector: Error setting data "
                f"at ({row},{col_idx}) to '{value_str}': {e}"
            )

    def add_surface(self, index: int = -1):
        """Adds a new, standard surface to the optical system.

        Args:
            index (int): The position in the Lens Data Editor where the surface
                         should be inserted. If -1, it is added before the image plane.
        """
        old_state = self._capture_optic_state()
        num_lde_rows = self.get_surface_count()
        optic_insert_idx = num_lde_rows - 1 if num_lde_rows > 1 else 1
        if index != -1 and 0 < index < num_lde_rows:
            optic_insert_idx = index
        params = {
            "surface_type": "standard",
            "radius": float("inf"),
            "thickness": 0.0,
            "material": "Air",
            "comment": "New Surface",
            "index": optic_insert_idx,
        }
        self._optic.add_surface(**params)
        self._optic.update()
        self._undo_redo_manager.add_state(old_state)
        self.set_modified(True)
        self.surfaceAdded.emit(optic_insert_idx)
        self.surfaceCountChanged.emit()
        self.opticChanged.emit()

    def remove_surface(self, lde_row_index: int):
        """
        Removes a surface from the system.
        Note: Cannot remove the Object (0) or Image surfaces.
        """
        old_state = self._capture_optic_state()
        optic_surface_index = lde_row_index
        if 0 < optic_surface_index < self.get_surface_count() - 1:
            self._optic.surface_group.remove_surface(optic_surface_index)
            self._optic.update()
            self._undo_redo_manager.add_state(old_state)
            self.set_modified(True)
            self.surfaceRemoved.emit(lde_row_index)
            self.surfaceCountChanged.emit()
            self.opticChanged.emit()
        else:
            print(
                "OpticConnector: Cannot remove Object or Image surface "
                "via this LDE action."
            )

    def undo(self):
        """Performs an undo operation, reverting to the previous system state."""
        if self._undo_redo_manager.can_undo():
            current_state_for_redo = self._capture_optic_state()
            restored_state_data = self._undo_redo_manager.undo(current_state_for_redo)
            if restored_state_data:
                self._restore_optic_state(restored_state_data)
                print("OpticConnector: Undo successful.")
            else:
                print("OpticConnector: Undo operation failed to return state data.")
        else:
            print("OpticConnector: Cannot undo.")

    def redo(self):
        """Performs a redo operation, reapplying an undone change."""
        if self._undo_redo_manager.can_redo():
            current_state_for_undo = self._capture_optic_state()
            restored_state_data = self._undo_redo_manager.redo(current_state_for_undo)
            if restored_state_data:
                self._restore_optic_state(restored_state_data)
                print("OpticConnector: Redo successful.")
            else:
                print("OpticConnector: Redo operation failed to return state data.")
        else:
            print("OpticConnector: Cannot redo.")

    def get_field_options(self):
        """
        Returns a list of (display_name, value_str) tuples for field settings.
        """
        if not self._optic or self._optic.fields.num_fields == 0:
            return [("all", "'all'")]

        options = [("all", "'all'")]

        field_coords = self._optic.fields.get_field_coords()

        for i, field_tuple in enumerate(field_coords):
            hx, hy = field_tuple[0], field_tuple[1]

            display_name = f"Field {i + 1}: ({hx:.3f}, {hy:.3f})"

            value_str = f"[({hx}, {hy})]"
            options.append((display_name, value_str))

        return options

    def get_wavelength_options(self):
        """
        Returns a list of (display_name, value_str) tuples for wavelength settings.
        """
        if not self._optic or self._optic.wavelengths.num_wavelengths == 0:
            return [("all", "'all'"), ("primary", "'primary'")]

        options = [("all", "'all'"), ("primary", "'primary'")]

        wavelength_values = self._optic.wavelengths.get_wavelengths()

        for wl_value in wavelength_values:
            display_name = f"{wl_value:.4f} µm"
            value_str = f"[{wl_value}]"
            options.append((display_name, value_str))

        return options
