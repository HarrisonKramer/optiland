"""
Zemax File Handler

This module provides functionality for reading Zemax files and converting the
data into Optiland Optic instances. The `load_zemax_file` function can be used
to load a Zemax file and return an Optiland Optic object.

Kramer Harrison, 2024
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any

import requests

import optiland.backend as be
from optiland.fileio.converters import ZemaxToOpticConverter
from optiland.materials import AbbeMaterial, BaseMaterial, Material


def load_zemax_file(source: str):
    """Loads a Zemax file and returns an Optic object.

    Args:
        source (str): The path to a local .zmx file or a URL pointing to a .zmx file.

    Returns:
        Optic: An `Optic` object created from the Zemax file data.
    """
    # 1. Resolve the file source
    source_handler = ZemaxFileSourceHandler(source)
    filename = source_handler.get_local_file()

    try:
        # 2. Parse file into structured data
        parser = ZemaxDataParser(filename)
        data_model = parser.parse()

        # 3. Convert into Optiland Optic
        converter = ZemaxToOpticConverter(data_model.to_dict())
        return converter.convert()

    finally:
        source_handler.cleanup()


@dataclass
class ZemaxDataModel:
    """Data model containing data extracted from Zemax files

    Contains data related to:
        - Aperture
        - Fields
        - Wavelengths
        - Surfaces & all related surface information
        - Glass Catalogs
    """

    aperture: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    wavelengths: dict[str, Any] = field(default_factory=lambda: {"data": []})
    surfaces: dict[int, dict[str, Any]] = field(default_factory=dict)
    glass_catalogs: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the data model as a plain dictionary."""
        result = {
            "aperture": self.aperture,
            "fields": self.fields,
            "wavelengths": self.wavelengths,
            "surfaces": self.surfaces,
        }
        if self.glass_catalogs is not None:
            result["glass_catalogs"] = self.glass_catalogs
        return result


class ZemaxFileSourceHandler:
    """Handles source input resolution for Zemax files (local vs URL)."""

    def __init__(self, source: str):
        self.source = source
        self._is_tempfile = False
        self._local_file: str | None = None

    def _is_url(self) -> bool:
        return re.match(r"^https?://", self.source) is not None

    def get_local_file(self) -> str:
        if self._is_url():
            response = requests.get(self.source, timeout=10)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(response.content)
                    self._local_file = tmp.name
                self._is_tempfile = True
            else:
                raise ValueError("Failed to download Zemax file.")
        else:
            self._local_file = self.source
        return self._local_file

    def cleanup(self):
        if self._is_tempfile and self._local_file and os.path.exists(self._local_file):
            os.remove(self._local_file)


class ZemaxDataParser:
    """Parses Zemax .zmx file into a ZemaxDataModel."""

    def __init__(self, filename: str):
        self.filename = filename
        self.data_model = ZemaxDataModel()
        self._current_surf = -1
        self._current_surf_data: dict[str, Any] = {}

        # Operand dispatch table
        self._operand_table = {
            "FNUM": self._read_fno,
            "ENPD": self._read_epd,
            "OBNA": self._read_object_na,
            "FLOA": self._read_floating_stop,
            "FTYP": self._read_config_data,
            "XFLN": self._read_x_fields,
            "YFLN": self._read_y_fields,
            "WAVM": self._read_wavelength,
            "PWAV": self._read_primary_wave,
            "SURF": self._read_surface,
            "TYPE": self._read_surf_type,
            "PARM": self._read_surface_parameter,
            "CURV": self._read_radius,
            "DISZ": self._read_thickness,
            "CONI": self._read_conic,
            "GLAS": self._read_glass,
            "STOP": self._read_stop,
            "MODE": self._read_mode,
            "GCAT": self._read_glass_catalog,
            "FWGN": self._read_field_weights,
            "VDXN": self._read_vignette_decenter_x,
            "VDYN": self._read_vignette_decenter_y,
            "VCXN": self._read_vignette_compress_x,
            "VCYN": self._read_vignette_compress_y,
            "VANN": self._read_vignette_tangent_angle,
        }

    def parse(self) -> ZemaxDataModel:
        """Reads the Zemax file and extracts the optical data into ZemaxDataModel."""
        encodings = ["utf-16", "utf-8", "iso-8859-1"]
        success = False
        for encoding in encodings:
            try:
                with open(self.filename, encoding=encoding) as file:
                    for line in file:
                        data = line.split()
                        if not data:
                            continue
                        operand = data[0]
                        if operand in self._operand_table:
                            self._operand_table[operand](data)
            except (UnicodeError, UnicodeDecodeError):
                continue

            if self.data_model.aperture:
                success = True
                break

        if not success:
            raise ValueError("Failed to read Zemax file.")

        self._finalize_fields()
        self._finalize_surface()
        return self.data_model

    # ------------------ Parsing methods ------------------
    def _read_fno(self, data):
        if int(data[2]) == 0:
            self.data_model.aperture["imageFNO"] = float(data[1])
        elif int(data[2]) == 1:
            self.data_model.aperture["paraxialImageFNO"] = float(data[1])

    def _read_epd(self, data):
        self.data_model.aperture["EPD"] = float(data[1])

    def _read_object_na(self, data):
        if int(data[2]) == 0:
            self.data_model.aperture["objectNA"] = float(data[1])
        elif int(data[2]) == 1:
            self.data_model.aperture["object_cone_angle"] = float(data[1])

    def _read_floating_stop(self, data):
        self.data_model.aperture["floating_stop"] = True

    def _read_config_data(self, data):
        fields = self.data_model.fields
        fields["num_fields"] = int(data[3])
        fields["type"] = {
            0: "angle",
            1: "object_height",
            2: "paraxial_image_height",
            3: "real_image_height",
            4: "theodolite_angle",
        }.get(int(data[1]), "unsupported")

        self.data_model.wavelengths["num_wavelengths"] = int(data[4])
        fields["object_space_telecentric"] = int(data[2]) == 1
        fields["afocal_image_space"] = int(data[7]) == 1

    def _read_x_fields(self, data):
        num_fields = self.data_model.fields["num_fields"]
        self.data_model.fields["x"] = [float(v) for v in data[1 : num_fields + 1]]

    def _read_y_fields(self, data):
        num_fields = self.data_model.fields["num_fields"]
        self.data_model.fields["y"] = [float(v) for v in data[1 : num_fields + 1]]

    def _read_wavelength(self, data):
        val = float(data[2])
        if (
            len(self.data_model.wavelengths["data"])
            < self.data_model.wavelengths["num_wavelengths"]
        ):
            self.data_model.wavelengths["data"].append(val)

    def _read_primary_wave(self, data):
        self.data_model.wavelengths["primary_index"] = int(data[1]) - 1

    def _read_surface(self, data):
        if self._current_surf >= 0:
            self.data_model.surfaces[self._current_surf] = self._current_surf_data
        self._current_surf += 1
        self._current_surf_data = {
            "type": "standard",
            "is_stop": False,
            "conic": 0.0,
            "material": "air",
        }

    def _read_radius(self, data):
        try:
            self._current_surf_data["radius"] = 1 / float(data[1])
        except ZeroDivisionError:
            self._current_surf_data["radius"] = be.inf

    def _read_thickness(self, data):
        if data[1] == "INFINITY":
            self._current_surf_data["thickness"] = be.inf
        else:
            self._current_surf_data["thickness"] = float(data[1])

    def _read_conic(self, data):
        self._current_surf_data["conic"] = float(data[1])

    def _read_glass(self, data):
        material = data[1]
        if material.upper() == "MIRROR":
            self._current_surf_data["material"] = "mirror"
            return

        self._current_surf_data["material"] = material
        try:
            self._current_surf_data["index"] = float(data[4].replace(",", "."))
            self._current_surf_data["abbe"] = float(data[5].replace(",", "."))
        except IndexError:
            self._current_surf_data["index"] = None
            self._current_surf_data["abbe"] = None

        try:
            self._current_surf_data["material"] = Material(material)
        except ValueError:
            if self.data_model.glass_catalogs:
                for mfg in self.data_model.glass_catalogs:
                    try:
                        self._current_surf_data["material"] = Material(
                            material, mfg.lower()
                        )
                        break
                    except ValueError:
                        continue

        if not isinstance(self._current_surf_data["material"], BaseMaterial):
            self._current_surf_data["material"] = AbbeMaterial(
                self._current_surf_data["index"], self._current_surf_data["abbe"]
            )

    def _read_stop(self, data):
        self._current_surf_data["is_stop"] = True

    def _read_mode(self, data):
        if data[1] != "SEQ":
            raise ValueError("Only sequential mode is supported.")

    def _read_glass_catalog(self, data):
        self.data_model.glass_catalogs = data[1:]

    def _read_surf_type(self, data):
        self._current_surf_data["type"] = {
            "STANDARD": "standard",
            "EVENASPH": "even_asphere",
            "ODDASPHE": "odd_asphere",
            "COORDBRK": "coordinate_break",
        }.get(data[1], "unsupported")

    def _read_surface_parameter(self, data):
        key = f"param_{int(data[1]) - 1}"
        self._current_surf_data[key] = float(data[2])

    def _read_field_weights(self, data):
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["weights"] = [float(v) for v in data[1 : n + 1]]

    def _read_vignette_decenter_x(self, data):
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["vignette_decenter_x"] = [
            float(v) for v in data[1 : n + 1]
        ]

    def _read_vignette_decenter_y(self, data):
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["vignette_decenter_y"] = [
            float(v) for v in data[1 : n + 1]
        ]

    def _read_vignette_compress_x(self, data):
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["vignette_compress_x"] = [
            float(v) for v in data[1 : n + 1]
        ]

    def _read_vignette_compress_y(self, data):
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["vignette_compress_y"] = [
            float(v) for v in data[1 : n + 1]
        ]

    def _read_vignette_tangent_angle(self, data):
        n = self.data_model.fields["num_fields"]
        self.data_model.fields["vignette_tangent_angle"] = [
            float(v) for v in data[1 : n + 1]
        ]

    # ------------------ Helpers ------------------
    def _finalize_fields(self):
        if "x" in self.data_model.fields and "y" in self.data_model.fields:
            unique_fields = set(
                zip(
                    self.data_model.fields["x"],
                    self.data_model.fields["y"],
                    strict=False,
                )
            )
            sorted_fields = sorted(unique_fields, key=lambda x: x[1])
            xs, ys = zip(*sorted_fields, strict=False)
            self.data_model.fields["x"], self.data_model.fields["y"] = xs, ys

    def _finalize_surface(self):
        if self._current_surf >= 0:
            self.data_model.surfaces[self._current_surf] = self._current_surf_data
