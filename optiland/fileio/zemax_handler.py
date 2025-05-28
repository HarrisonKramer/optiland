"""Zemax File Handler

This module provides functionality for reading Zemax files and converting the
data into Optiland Optic instances. The `load_zemax_file` function can be used
to load a Zemax file and return an Optiland Optic object.

Kramer Harrison, 2024
"""

import os
import re
import tempfile

import requests

import optiland.backend as be
from optiland.fileio.converters import ZemaxToOpticConverter
from optiland.materials import AbbeMaterial, BaseMaterial, Material


def load_zemax_file(source):
    """Loads a Zemax file and returns an Optic object.

    Args:
        source (str): The path to a local .zmx file or a URL pointing to a
            .zmx file.

    Returns:
        Optic: An `Optic` object created from the Zemax file data.

    """
    reader = ZemaxFileReader(source)
    return reader.generate_lens()


class ZemaxFileReader:
    """A class for reading Zemax files and extracting optical data.

    Args:
        source (str): The path to a local .zmx file or a URL pointing to a
            .zmx file.

    Attributes:
        source (str): The original source string (filepath or URL).
        filename (str): The local path to the Zemax file (this might be a
            temporary file if the source was a URL).
        data (dict): A nested dictionary storing the extracted optical data,
            organized into keys like 'aperture', 'fields', 'wavelengths',
            and 'surfaces'.
        _current_surf_data (dict): Temporary storage for data of the surface
            currently being processed.
        _operand_table (dict): A mapping from Zemax operand codes (e.g.,
            "CURV", "DISZ") to their corresponding parsing methods in this class.
        _current_surf (int): The index of the surface currently being processed
            from the Zemax file.

    Methods:
        generate_lens(): Converts the extracted data into an Optiland optic
            instance.

    """

    def __init__(self, source):
        self.source = source

        self.data = {}
        self.data["aperture"] = {}
        self.data["fields"] = {}
        self.data["wavelengths"] = {}
        self.data["wavelengths"]["data"] = []
        self.data["surfaces"] = {}

        self._current_surf_data = {}

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

        self._current_surf = -1
        self._configure_source_input()
        self._read_file()

    def generate_lens(self):
        """Converts the extracted optical data into an Optiland `Optic` instance.

        Returns:
            Optic: The generated `Optic` object.
        """
        converter = ZemaxToOpticConverter(self.data)
        return converter.convert()

    def _is_url(self, source: str) -> bool:
        """Check if the source string is a URL.

        Args:
            source (str): The source to check.

        Returns:
            bool: True if the source is a URL, False otherwise.

        """
        return re.match(r"^https?://", source) is not None

    def _configure_source_input(self):
        """Checks if the source is a URL and writes to a temporary file if so.
        Otherwise, sets the source to the filename.
        """
        if self._is_url(self.source):
            response = requests.get(self.source)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False) as file:
                    file.write(response.content)
                    self.filename = file.name
            else:
                raise ValueError("Failed to download Zemax file.")
        else:
            self.filename = self.source

    def _read_file(self):
        """Reads the Zemax file and extracts the optical data."""
        encodings = ["utf-16", "utf-8", "iso-8859-1"]
        success = False
        for encoding in encodings:
            try:
                with open(self.filename, encoding=encoding) as file:
                    for line in file:
                        data = line.split()
                        try:
                            operand = data[0]
                            self._operand_table[operand](data)
                        except (IndexError, KeyError):
                            continue
            except (UnicodeError, UnicodeDecodeError):
                continue

            if self.data["aperture"]:
                success = True
                break
            else:
                continue

        if not success:
            raise ValueError("Failed to read Zemax file.")

        # sort and filter fields
        unique_fields = set()
        for i in range(
            min(len(self.data["fields"]["x"]), len(self.data["fields"]["y"])),
        ):
            pair = (self.data["fields"]["x"][i], self.data["fields"]["y"][i])
            unique_fields.add(pair)

        # Sort the unique field pairs based on the second element
        sorted_fields = sorted(unique_fields, key=lambda x: x[1])

        # Unzip the sorted pairs back into two lists for x, y fields
        self.data["fields"]["x"], self.data["fields"]["y"] = zip(*sorted_fields)

        # remove temporary file if it was created
        if self._is_url(self.source):
            os.remove(self.filename)

    def _read_fno(self, data):
        """Extracts the FNO (F-number) data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        if int(data[2]) == 0:
            self.data["aperture"]["imageFNO"] = float(data[1])
        elif int(data[2]) == 1:
            self.data["aperture"]["paraxialImageFNO"] = float(data[1])

    def _read_epd(self, data: list[str]):
        """Extracts the EPD (Entrance Pupil Diameter) data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        self.data["aperture"]["EPD"] = float(data[1])

    def _read_object_na(self, data: list[str]):
        """Extracts the object-space numerical aperture (NA).

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        if int(data[2]) == 0:
            self.data["aperture"]["objectNA"] = float(data[1])
        elif int(data[2]) == 1:
            self.data["aperture"]["object_cone_angle"] = float(data[1])

    def _read_floating_stop(self, data: list[str]):
        """Marks the aperture definition as 'floating_stop'.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        self.data["aperture"]["floating_stop"] = True

    def _read_config_data(self, data: list[str]):
        """Extracts general configuration data like field type, number of
        wavelengths, etc.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        self.data["fields"]["num_fields"] = int(data[3])

        if int(data[1]) == 0:
            self.data["fields"]["type"] = "angle"
        elif int(data[1]) == 1:
            self.data["fields"]["type"] = "object_height"
        elif int(data[1]) == 2:
            self.data["fields"]["type"] = "paraxial_image_height"
        elif int(data[1]) == 3:
            self.data["fields"]["type"] = "real_image_height"
        elif int(data[1]) == 4:
            self.data["fields"]["type"] = "theodolite_angle"
        else:
            self.data["fields"]["type"] = "unsupported"

        self.data["wavelengths"]["num_wavelengths"] = int(data[4])

        if int(data[2]) == 1:
            self.data["fields"]["object_space_telecentric"] = True
        else:
            self.data["fields"]["object_space_telecentric"] = False

        if int(data[7]) == 1:
            self.data["fields"]["afocal_image_space"] = True
        else:
            self.data["fields"]["afocal_image_space"] = False

    def _read_x_fields(self, data):
        """Extracts the x-field data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        num_fields = self.data["fields"]["num_fields"]
        self.data["fields"]["x"] = [float(value) for value in data[1 : num_fields + 1]]

    def _read_y_fields(self, data: list[str]):
        """Extracts the Y-field data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        num_fields = self.data["fields"]["num_fields"]
        self.data["fields"]["y"] = [float(value) for value in data[1 : num_fields + 1]]

    def _read_wavelength(self, data: list[str]):
        """Extracts wavelength data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        value = float(data[2])
        num_wavelengths = self.data["wavelengths"]["num_wavelengths"]
        if len(self.data["wavelengths"]["data"]) < num_wavelengths:
            self.data["wavelengths"]["data"].append(value)

    def _read_surface(self, data: list[str]):
        """Initializes data storage for a new surface.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        if self._current_surf >= 0:
            self.data["surfaces"][self._current_surf] = self._current_surf_data

        self._current_surf_data = {}
        self._current_surf_data["type"] = "standard"  # Default type
        self._current_surf_data["is_stop"] = False
        self._current_surf_data["conic"] = 0.0
        self._current_surf_data["material"] = "air"  # Default material
        self._current_surf += 1

    def _read_radius(self, data: list[str]):
        """Extracts the surface radius data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        try:
            self._current_surf_data["radius"] = 1 / float(data[1])
        except ZeroDivisionError:
            self._current_surf_data["radius"] = be.inf  # Use backend.inf

    def _read_thickness(self, data: list[str]):
        """Extracts the surface thickness data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        if data[1] == "INFINITY":
            self._current_surf_data["thickness"] = be.inf  # Use backend.inf
        else:
            self._current_surf_data["thickness"] = float(data[1])

    def _read_conic(self, data: list[str]):
        """Extracts the surface conic constant data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        self._current_surf_data["conic"] = float(data[1])

    def _read_glass(self, data: list[str]):
        """Extracts the surface material (glass) data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.
                Example for mirror: `['GLAS', 'MIRROR', '0', '0', '1.5', '40', ...]`

        """
        material = data[1]

        # for now we consider that there are no glasses directly
        # after a mirror. This is not always true. Later we have
        # to account for it too.
        if material.upper() == "MIRROR":
            self._current_surf_data["material"] = "mirror"
            return
        self._current_surf_data["material"] = material
        self._current_surf_data["index"] = float(data[4].replace(",", "."))
        self._current_surf_data["abbe"] = float(data[5].replace(",", "."))

        # Generate a Material object from the material name & manufacturer
        try:
            # Try to create a Material object from the material name
            self._current_surf_data["material"] = Material(material)
        except ValueError:
            # If the material name is not recognized, try to create a Material
            # object from the material name and manufacturer
            if "glass_catalogs" in self.data:
                for manufacturer in self.data["glass_catalogs"]:
                    try:
                        self._current_surf_data["material"] = Material(
                            material,
                            manufacturer.lower(),
                        )
                        break
                    except ValueError:
                        continue

        # If the material is still not recognized, use model glass
        if not isinstance(self._current_surf_data["material"], BaseMaterial):
            n = self._current_surf_data["index"]
            v = self._current_surf_data["abbe"]
            self._current_surf_data["material"] = AbbeMaterial(n, v)

    def _read_stop(self, data):
        """Extracts the stop data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        self._current_surf_data["is_stop"] = True

    def _read_primary_wave(self, data: list[str]):
        """Extracts the primary wavelength index.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        self.data["wavelengths"]["primary_index"] = int(data[1]) - 1

    def _read_mode(self, data: list[str]):
        """Extracts the ray tracing mode (sequential or non-sequential).

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        Raises:
            ValueError: If the mode is not sequential.
        """
        if data[1] != "SEQ":
            raise ValueError("Only sequential mode is supported.")

    def _read_glass_catalog(self, data: list[str]):
        """Extracts the list of used glass catalogs.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        self.data["glass_catalogs"] = data[1:]

    def _read_surf_type(self, data: list[str]):
        """Extracts the surface type (e.g., Standard, Even Asphere).

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        type_map = {
            "STANDARD": "standard",
            "EVENASPH": "even_asphere",
            "ODDASPHE": "odd_asphere",
            "COORDBRK": "coordinate_break",
        }
        try:
            self._current_surf_data["type"] = type_map[data[1]]
        except KeyError:
            self._current_surf_data["type"] = "unsupported"

    def _read_surface_parameter(self, data):
        """Extracts the surface parameter data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        key = f"param_{int(data[1]) - 1}"  # Zemax PARM is 1-indexed
        self._current_surf_data[key] = float(data[2])

    def _read_field_weights(self, data: list[str]):
        """Extracts the field weights data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        num_fields = self.data["fields"]["num_fields"]
        self.data["fields"]["weights"] = [
            float(value) for value in data[1 : num_fields + 1]
        ]

    def _read_vignette_decenter_x(self, data: list[str]):
        """Extracts the vignette decenter X data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        num_fields = self.data["fields"]["num_fields"]
        self.data["fields"]["vignette_decenter_x"] = [
            float(value) for value in data[1 : num_fields + 1]
        ]

    def _read_vignette_decenter_y(self, data: list[str]):
        """Extracts the vignette decenter Y data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        num_fields = self.data["fields"]["num_fields"]
        self.data["fields"]["vignette_decenter_y"] = [
            float(value) for value in data[1 : num_fields + 1]
        ]

    def _read_vignette_compress_x(self, data: list[str]):
        """Extracts the vignette compress X data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        num_fields = self.data["fields"]["num_fields"]
        self.data["fields"]["vignette_compress_x"] = [
            float(value) for value in data[1 : num_fields + 1]
        ]

    def _read_vignette_compress_y(self, data: list[str]):
        """Extracts the vignette compress Y data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        num_fields = self.data["fields"]["num_fields"]
        self.data["fields"]["vignette_compress_y"] = [
            float(value) for value in data[1 : num_fields + 1]
        ]

    def _read_vignette_tangent_angle(self, data: list[str]):
        """Extracts the vignette tangent angle data.

        Args:
            data (list[str]): List of string data values from the Zemax file line.

        """
        num_fields = self.data["fields"]["num_fields"]
        self.data["fields"]["vignette_tangent_angle"] = [
            float(value) for value in data[1 : num_fields + 1]
        ]
