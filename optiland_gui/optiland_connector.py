"""Defines the connector that bridges the GUI and the Optiland core logic.

This module contains the `OptilandConnector` class, which acts as a vital
intermediary between the user interface and the underlying `optiland` optical
engine. It manages the active optical system (`Optic` object), handles file
I/O, provides data to UI components like the Lens Editor, and manages the
undo/redo stack.

Author: Manuel Fragata Mendes, 2025
"""

import ast
import json

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.materials import Material as OptilandMaterial
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.physical_apertures.radial import configure_aperture
from optiland.surfaces.factories.geometry_factory import GeometryConfig, GeometryFactory
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
    """The main bridge between the Optiland core logic and the GUI."""

    opticLoaded = Signal()
    opticChanged = Signal()
    modifiedStateChanged = Signal(bool)
    surfaceDataChanged = Signal(int, int, object)
    surfaceAdded = Signal(int)
    surfaceRemoved = Signal(int)
    surfaceCountChanged = Signal()
    undoStackAvailabilityChanged = Signal(bool)
    redoStackAvailabilityChanged = Signal(bool)

    # Column indices
    COL_TYPE = 0
    COL_COMMENT = 1
    COL_RADIUS = 2
    COL_THICKNESS = 3
    COL_MATERIAL = 4
    COL_CONIC = 5
    COL_SEMI_DIAMETER = 6

    DEFAULT_WAVELENGTH_UM = 0.550

    AVAILABLE_SURFACE_TYPES = [
        "standard",
        "paraxial",
        "biconic",
        "chebyshev",
        "even_asphere",
        "odd_asphere",
        "polynomial",
        "toroidal",
        "zernike",
    ]

    # Map for EXTRA parameters that appear ONLY in the properties box
    EXTRA_PARAM_MAP = {
        "BiconicGeometry": {
            "Radius X": "Rx",
            "Conic X": "kx",
        },
        "ChebyshevPolynomialGeometry": {
            "Coefficients": "c",
            "Norm X": "norm_x",
            "Norm Y": "norm_y",
        },
        "EvenAsphere": {"Coefficients": "c"},
        "OddAsphere": {"Coefficients": "c"},
        "PolynomialGeometry": {"Coefficients": "c"},
        "ToroidalGeometry": {
            "Radius of Rotation (XZ)": "R_rot",
            "Coefficients YZ": "coeffs_poly_y",
        },
        "ZernikePolynomialGeometry": {
            "Coefficients": "c",
            "Normalization Radius": "norm_radius",
        },
    }

    def __init__(self):
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
        self.opticChanged.emit()
        self._undo_redo_manager.clear_stacks()

    def set_modified(self, modified: bool):
        if self._is_modified != modified:
            self._is_modified = modified
            self.modifiedStateChanged.emit(self._is_modified)

    def is_modified(self) -> bool:
        return self._is_modified

    def _initialize_optic_structure(
        self, optic_instance: Optic, is_specific_new_system: bool = False
    ):
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
            optic_instance.set_field_type("angle")
            optic_instance.add_field(y=0)
            # set system aperture
            optic_instance.pupil_type = "EPD"
            optic_instance.entrance_pupil_diameter = 10.0
        else:
            if optic_instance.surface_group.num_surfaces < 2:
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
                optic_instance.add_wavelength(
                    self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um"
                )
            elif (
                optic_instance.wavelengths.primary_index is None
                and optic_instance.wavelengths.num_wavelengths > 0
            ):
                optic_instance.wavelengths.wavelengths[0].is_primary = True
        optic_instance.update()

    def _get_safe_primary_wavelength_value(self) -> float:
        if self._optic.wavelengths.num_wavelengths > 0:
            primary_idx = self._optic.wavelengths.primary_index
            if primary_idx is not None:
                try:
                    return self._optic.wavelengths.wavelengths[primary_idx].value
                except IndexError:
                    pass
            if self._optic.wavelengths.num_wavelengths > 0:
                self._optic.wavelengths.wavelengths[0].is_primary = True
                return self._optic.wavelengths.wavelengths[0].value
        if self._optic.wavelengths.num_wavelengths == 0:
            self._optic.add_wavelength(
                self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um"
            )
            self._optic.update()
        return self.DEFAULT_WAVELENGTH_UM

    def get_optic(self) -> Optic:
        return self._optic

    def _capture_optic_state(self):
        if self._optic.wavelengths.num_wavelengths == 0:
            self._optic.add_wavelength(
                self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um"
            )
        elif (
            self._optic.wavelengths.primary_index is None
            and self._optic.wavelengths.num_wavelengths > 0
        ):
            self._optic.wavelengths.wavelengths[0].is_primary = True
        self._optic.update()
        return self._optic.to_dict()

    def _restore_optic_state(self, state_data):
        self._optic = Optic.from_dict(state_data)
        self._initialize_optic_structure(self._optic, is_specific_new_system=False)
        self.opticLoaded.emit()

    def new_system(self):
        self._undo_redo_manager.clear_stacks()
        self._optic = Optic("New Untitled System")
        self._initialize_optic_structure(self._optic, is_specific_new_system=True)
        self._current_filepath = None
        self.set_modified(False)
        self.opticLoaded.emit()
        self.opticChanged.emit()

    def load_optic_from_file(self, filepath: str):
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
            self._initialize_optic_structure(self._optic, is_specific_new_system=False)
            self.set_modified(False)
            self.opticLoaded.emit()
        except Exception as e:
            QMessageBox.critical(
                None, "Load Error", f"Failed to load system from {filepath}:\n{e}"
            )
            self.new_system()

    def save_optic_to_file(self, filepath: str):
        try:
            data = self._capture_optic_state()
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4, cls=SpecialFloatEncoder)
            self._current_filepath = filepath
            self.set_modified(False)
        except Exception as e:
            QMessageBox.critical(
                None, "Save Error", f"Could not save system to {filepath}:\n{e}"
            )

    def get_current_filepath(self) -> str | None:
        return self._current_filepath

    def get_surface_count(self) -> int:
        return (
            self._optic.surface_group.num_surfaces
            if self._optic and self._optic.surface_group
            else 0
        )

    def get_column_headers(self, row=-1) -> list[str]:
        """Returns column headers, dynamically changing for the selected row
        if provided."""
        headers = [
            "Type",
            "Comment",
            "Radius",
            "Thickness",
            "Material",
            "Conic",
            "Semi-Diameter",
        ]
        if not (0 <= row < self.get_surface_count()):
            return headers

        surface = self._optic.surface_group.surfaces[row]
        if surface.surface_type == "paraxial":
            headers[self.COL_RADIUS] = "Focal Length"
            headers[self.COL_CONIC] = "N/A"
        elif surface.surface_type == "toroidal":
            headers[self.COL_RADIUS] = "Radius YZ"
            headers[self.COL_CONIC] = "Conic YZ"
        elif surface.surface_type == "biconic":
            headers[self.COL_RADIUS] = "Radius Y"
            headers[self.COL_CONIC] = "Conic Y"
        return headers

    def get_available_surface_types(self) -> list[str]:
        return self.AVAILABLE_SURFACE_TYPES

    def get_surface_type_info(self, row: int) -> dict:
        if not (0 <= row < self.get_surface_count()):
            return {"display_text": "Error", "is_changeable": False}
        surface = self._optic.surface_group.surfaces[row]
        if row == 0:
            return {"display_text": "Object", "is_changeable": False}
        if row == self.get_surface_count() - 1:
            return {"display_text": "Image", "is_changeable": False}
        base_type = surface.surface_type or "Standard"
        display_text = (
            f"Stop ({base_type.title()})" if surface.is_stop else base_type.title()
        )
        has_extra_params = bool(self.get_surface_geometry_params(row))
        return {
            "display_text": display_text,
            "is_changeable": not surface.is_stop,
            "has_extra_params": has_extra_params,
        }

    def get_surface_data(self, row: int, col_idx: int):
        if not (0 <= row < self.get_surface_count()):
            return None
        surface = self._optic.surface_group.surfaces[row]

        def format_val(val):
            return "inf" if val == float("inf") else f"{val:.4f}"

        if col_idx == self.COL_TYPE:
            return self.get_surface_type_info(row)["display_text"]
        if col_idx == self.COL_COMMENT:
            return surface.comment
        if col_idx == self.COL_THICKNESS:
            if row < self.get_surface_count() - 1:
                t = self._optic.surface_group.get_thickness(row)
                return f"{float(t[0]):.4f}" if t is not None and len(t) > 0 else "N/A"
            return "N/A"
        if col_idx == self.COL_MATERIAL:
            if surface.is_reflective:
                return "Mirror"
            mat = surface.material_post
            if isinstance(mat, IdealMaterial):
                n = mat.n(self._get_safe_primary_wavelength_value())
                return "Air" if n == 1.0 else f"Ideal n={n:.4f}"
            return mat.name if isinstance(mat, OptilandMaterial) else "Unknown"
        if col_idx == self.COL_SEMI_DIAMETER:
            if isinstance(surface.aperture, RadialAperture):
                return f"{float(surface.aperture.r_max):.4f}"
            if surface.semi_aperture is not None:
                return f"{float(surface.semi_aperture):.4f}"
            return "Auto"

        # Dynamic columns (Radius/Conic)
        geo = surface.geometry
        if surface.surface_type == "paraxial":
            if col_idx == self.COL_RADIUS:
                return format_val(surface.f)
            if col_idx == self.COL_CONIC:
                return "N/A"
        elif surface.surface_type == "toroidal":
            if col_idx == self.COL_RADIUS:
                return format_val(getattr(geo, "R_yz", float("inf")))
            if col_idx == self.COL_CONIC:
                return format_val(getattr(geo, "k_yz", 0.0))
        elif surface.surface_type == "biconic":
            if col_idx == self.COL_RADIUS:
                return format_val(getattr(geo, "Ry", float("inf")))
            if col_idx == self.COL_CONIC:
                return format_val(getattr(geo, "ky", 0.0))

        # Default Radius/Conic for all other types (including Chebyshev)
        if col_idx == self.COL_RADIUS:
            return format_val(geo.radius)
        if col_idx == self.COL_CONIC:
            return format_val(geo.k if hasattr(geo, "k") else 0.0)

        return None

    def set_surface_type(self, row: int, new_type: str):
        if not (0 < row < self.get_surface_count() - 1):
            return
        new_type = new_type.lower().strip()
        if new_type not in self.AVAILABLE_SURFACE_TYPES:
            return

        old_state = self._capture_optic_state()
        try:
            surface = self._optic.surface_group.surfaces[row]
            old_geo = surface.geometry
            config = GeometryConfig(
                radius=getattr(old_geo, "radius", float("inf")),
                conic=getattr(old_geo, "k", 0.0),
            )

            # Special handling for each surface type to ensure proper geometry creation
            if new_type == "biconic":
                # Extract existing biconic parameters if converting from biconic,
                # otherwise use finite defaults
                if hasattr(old_geo, "Rx") and hasattr(old_geo, "Ry"):
                    config.radius_x = getattr(old_geo, "Rx", 100.0)
                    config.radius_y = getattr(old_geo, "Ry", 100.0)
                    config.conic_x = getattr(old_geo, "kx", 0.0)
                    config.conic_y = getattr(old_geo, "ky", 0.0)
                else:
                    # Use finite default values to ensure BiconicGeometry is created,
                    # not Plane
                    config.radius_x = 100.0  # Finite default to avoid Plane creation
                    config.radius_y = 100.0  # Finite default to avoid Plane creation
                    config.conic_x = 0.0
                    config.conic_y = 0.0

            elif new_type == "toroidal":
                # Extract existing toroidal parameters if available
                if hasattr(old_geo, "R_rot"):
                    config.radius = getattr(old_geo, "R_rot", 100.0)  # Rotation radius
                    config.radius_y = getattr(old_geo, "R_yz", 100.0)  # YZ radius
                    config.conic = getattr(old_geo, "k_yz", 0.0)  # YZ conic
                    config.toroidal_coeffs_poly_y = getattr(
                        old_geo, "coeffs_poly_y", []
                    )
                else:
                    config.radius = 100.0  # Finite default for rotation radius
                    config.radius_y = 100.0  # Finite default for YZ radius
                    config.conic = 0.0
                    config.toroidal_coeffs_poly_y = []

            elif new_type in ["even_asphere", "odd_asphere", "polynomial"]:
                # Extract existing coefficients if available
                if hasattr(old_geo, "c"):
                    config.coefficients = getattr(old_geo, "c", [])
                else:
                    config.coefficients = []
                # Use finite default radius to avoid Plane creation
                if config.radius == float("inf"):
                    config.radius = 100.0

            elif new_type == "chebyshev":
                # Extract existing Chebyshev parameters if available
                if hasattr(old_geo, "c"):
                    config.coefficients = getattr(old_geo, "c", [])
                    config.norm_x = getattr(old_geo, "norm_x", 1.0)
                    config.norm_y = getattr(old_geo, "norm_y", 1.0)
                else:
                    config.coefficients = []
                    config.norm_x = 1.0
                    config.norm_y = 1.0
                # Use finite default radius to avoid Plane creation
                if config.radius == float("inf"):
                    config.radius = 100.0

            elif new_type == "zernike":
                # Extract existing Zernike parameters if available
                if hasattr(old_geo, "c"):
                    config.coefficients = getattr(old_geo, "c", [])
                    config.norm_radius = getattr(old_geo, "norm_radius", 1.0)
                else:
                    config.coefficients = []
                    config.norm_radius = 1.0
                # Use finite default radius to avoid Plane creation
                if config.radius == float("inf"):
                    config.radius = 100.0

            elif new_type == "standard" and config.radius == float("inf"):
                # For standard surfaces, use finite default if radius is infinite
                config.radius = 100.0

            # paraxial surfaces are handled as Plane geometries,
            # so no special config needed

            new_geo = GeometryFactory.create(
                surface_type=new_type, cs=old_geo.cs, config=config
            )

            surface.geometry = new_geo
            surface.surface_type = new_type
            if new_type == "paraxial" and not hasattr(surface, "f"):
                surface.f = float("inf")  # Default focal length for new paraxial

            self._optic.update()
            self._undo_redo_manager.add_state(old_state)
            self.set_modified(True)
            self.opticChanged.emit()
        except Exception:
            self._restore_optic_state(old_state)

    def set_surface_data(self, row: int, col_idx: int, value_str: str):
        if not (0 <= row < self.get_surface_count()):
            return
        old_state = self._capture_optic_state()
        try:
            surface = self._optic.surface_group.surfaces[row]
            updater = self._optic._updater

            if col_idx == self.COL_COMMENT:
                surface.comment = value_str
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
            else:
                # Handle all other columns which are numeric
                val = float("inf") if value_str.lower() == "inf" else float(value_str)
                if col_idx == self.COL_THICKNESS and row < self.get_surface_count() - 1:
                    updater.set_thickness(val, row)
                elif col_idx == self.COL_SEMI_DIAMETER:
                    try:
                        surface.aperture = configure_aperture(float(value_str) * 2.0)
                    except ValueError:
                        surface.aperture = None
                elif surface.surface_type == "paraxial" and col_idx == self.COL_RADIUS:
                    surface.f = val
                elif surface.surface_type == "toroidal":
                    if col_idx == self.COL_RADIUS:
                        surface.geometry.R_yz = be.array(val)
                        surface.geometry.c_yz = 1.0 / val if val != 0 else 0.0
                    elif col_idx == self.COL_CONIC:
                        surface.geometry.k_yz = be.array(val)
                elif surface.surface_type == "biconic":
                    if col_idx == self.COL_RADIUS:
                        surface.geometry.Ry = be.array(val)
                        surface.geometry.cy = 1.0 / val if val != 0 else 0.0
                    elif col_idx == self.COL_CONIC:
                        surface.geometry.ky = be.array(val)
                elif col_idx == self.COL_RADIUS:
                    updater.set_radius(val, row)
                elif col_idx == self.COL_CONIC and hasattr(surface.geometry, "k"):
                    updater.set_conic(val, row)

            self._optic.update()
            self._undo_redo_manager.add_state(old_state)
            self.set_modified(True)
            self.opticChanged.emit()
        except Exception as e:
            print(
                f"OpticConnector: Error setting data at ({row},{col_idx}) to "
                f"'{value_str}': {e}"
            )
            self._restore_optic_state(old_state)

    def add_surface(self, index: int = -1):
        old_state = self._capture_optic_state()
        num_rows = self.get_surface_count()
        insert_idx = num_rows - 1 if index == -1 or index >= num_rows else index
        if insert_idx <= 0:
            insert_idx = 1

        self._optic.add_surface(
            surface_type="standard",
            radius=float("inf"),
            thickness=0.0,
            material="Air",
            comment="New Surface",
            index=insert_idx,
        )
        self._optic.update()
        self._undo_redo_manager.add_state(old_state)
        self.set_modified(True)
        self.opticChanged.emit()

    def remove_surface(self, lde_row_index: int):
        if not (0 < lde_row_index < self.get_surface_count() - 1):
            print("OpticConnector: Cannot remove Object or Image surface.")
            return

        old_state = self._capture_optic_state()
        try:
            self._optic.surface_group.remove_surface(lde_row_index)
            self._optic.update()
            self._undo_redo_manager.add_state(old_state)
            self.set_modified(True)
            self.opticChanged.emit()
        except Exception:
            self._restore_optic_state(old_state)

    def undo(self):
        if self._undo_redo_manager.can_undo():
            state = self._undo_redo_manager.undo(self._capture_optic_state())
            if state:
                self._restore_optic_state(state)

    def redo(self):
        if self._undo_redo_manager.can_redo():
            state = self._undo_redo_manager.redo(self._capture_optic_state())
            if state:
                self._restore_optic_state(state)

    def get_surface_geometry_params(self, row: int) -> dict:
        if not (0 < row < self.get_surface_count() - 1):
            return {}
        surface = self._optic.surface_group.surfaces[row]
        geometry = surface.geometry
        geo_class_name = geometry.__class__.__name__

        # Special case: if surface_type is biconic but geometry is Plane,
        # show biconic params anyway
        if surface.surface_type == "biconic" and geo_class_name == "Plane":
            # Return default biconic X-direction parameters that can be edited
            # (Y-direction parameters are already in the main LDE table)
            return {
                "Radius X": float("inf"),
                "Conic X": 0.0,
            }

        params = {}
        if geo_class_name in self.EXTRA_PARAM_MAP:
            for label, attr_name in self.EXTRA_PARAM_MAP[geo_class_name].items():
                if hasattr(geometry, attr_name):
                    params[label] = getattr(geometry, attr_name)
        return params

    def set_surface_geometry_params(self, row: int, params_dict: dict):
        if not (0 < row < self.get_surface_count() - 1):
            return
        old_state = self._capture_optic_state()
        try:
            surface = self._optic.surface_group.surfaces[row]
            geometry = surface.geometry
            geo_class_name = geometry.__class__.__name__

            # Special case: if surface_type is biconic but geometry is Plane,
            # we need to create a new BiconicGeometry
            if surface.surface_type == "biconic" and geo_class_name == "Plane":
                # For biconic surfaces, Y-direction values come from the main table,
                # X-direction values come from the properties panel
                radius_x = float("inf")
                conic_x = 0.0

                # Get Y-direction values from the current geometry (main table values)
                radius_y = getattr(geometry, "radius", float("inf"))
                conic_y = getattr(geometry, "k", 0.0)

                for label, value_str in params_dict.items():
                    try:
                        if isinstance(value_str, str) and (
                            value_str.strip().startswith("[")
                            or value_str.strip().startswith("(")
                        ):
                            new_value = ast.literal_eval(value_str)
                        else:
                            new_value = float(value_str)

                        if label == "Radius X":
                            radius_x = new_value
                        elif label == "Conic X":
                            conic_x = new_value
                    except (ValueError, SyntaxError):
                        continue

                # Create new BiconicGeometry with the parameters
                from optiland.geometries.biconic import BiconicGeometry

                new_geometry = BiconicGeometry(
                    coordinate_system=geometry.cs,
                    radius_x=(
                        radius_x if not be.isinf(radius_x) else 50.0
                    ),  # Use finite default to avoid Plane
                    radius_y=(
                        radius_y if not be.isinf(radius_y) else 50.0
                    ),  # Use finite default to avoid Plane
                    conic_x=conic_x,
                    conic_y=conic_y,
                )

                surface.geometry = new_geometry
                geometry = new_geometry
                geo_class_name = geometry.__class__.__name__

            if geo_class_name in self.EXTRA_PARAM_MAP:
                for label, value_str in params_dict.items():
                    attr_name = self.EXTRA_PARAM_MAP[geo_class_name].get(label)
                    if attr_name and hasattr(geometry, attr_name):
                        try:
                            if isinstance(value_str, str) and (
                                value_str.strip().startswith("[")
                                or value_str.strip().startswith("(")
                            ):
                                new_value = ast.literal_eval(value_str)
                            else:
                                new_value = float(value_str)

                            setattr(geometry, attr_name, be.array(new_value))

                            # FIX: Sync parent/dependent attributes for specific
                            # geometries
                            if (
                                geo_class_name == "ToroidalGeometry"
                                and attr_name == "R_rot"
                            ):
                                geometry.radius = be.array(new_value)
                            if geo_class_name == "BiconicGeometry":
                                if attr_name == "Rx":
                                    geometry.radius = be.array(new_value)
                                    geometry.cx = be.where(
                                        be.isinf(new_value) | (new_value == 0),
                                        0.0,
                                        1.0 / new_value,
                                    )
                                elif attr_name == "Ry":
                                    geometry.cy = be.where(
                                        be.isinf(new_value) | (new_value == 0),
                                        0.0,
                                        1.0 / new_value,
                                    )
                                elif attr_name == "kx":
                                    geometry.k = be.array(new_value)

                        except (ValueError, SyntaxError):
                            continue

            self._optic.update()
            self._undo_redo_manager.add_state(old_state)
            self.set_modified(True)
            self.opticChanged.emit()
        except Exception:
            self._restore_optic_state(old_state)

    # --- Methods to fix crash ---
    def get_field_options(self):
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
        if not self._optic or self._optic.wavelengths.num_wavelengths == 0:
            return [("all", "'all'"), ("primary", "'primary'")]
        options = [("all", "'all'"), ("primary", "'primary'")]
        wavelength_values = self._optic.wavelengths.get_wavelengths()
        for wl_value in wavelength_values:
            display_name = f"{wl_value:.4f} µm"
            value_str = f"[{wl_value}]"
            options.append((display_name, value_str))
        return options
