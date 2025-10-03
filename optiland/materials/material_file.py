"""Material File

This module contains a class for representing a material based on a material
YAML file from the refractiveindex.info database.

Kramer Harrison, 2024
"""

from __future__ import annotations

import contextlib
import os
from io import StringIO

import numpy as np
import yaml

import optiland.backend as be
from optiland.environment.air_index import refractive_index_air
from optiland.environment.conditions import EnvironmentalConditions
from optiland.materials.base import BaseMaterial


class MaterialFile(BaseMaterial):
    """Represents a material based on a material YAML file from the
    refractiveindex.info database.

    Material refractive indices are based on various dispersion formulas or
    tabulated data. The material file contains the coefficients for the
    dispersion formulas and/or tabulated data.

    See https://refractiveindex.info/database/doc/Dispersion%20formulas.pdf

    Args:
        filename (str): The path to the material file.
        is_relative_to_air (bool): If True, the refractive index is calculated
            relative to air at standard conditions (15°C, 1 atm). If False, the
            refractive index is calculated relative to vacuum. Default is True.

    Attributes:
        filename (str): The filename of the material file.
        coefficients (list): A list of coefficients for calculating the
            refractive index.

    Methods:
        n(wavelength, temperature, pressure): Calculates the refractive index of the
            material at a given wavelength, temperature and pressure.
        k(wavelength): Retrieves the extinction coefficient of the material at
            a given wavelength.

    """

    def __init__(self, filename, is_relative_to_air: bool = True):
        super().__init__()
        self.filename = filename
        self.is_relative_to_air = is_relative_to_air
        self._k_warning_printed = False
        self.coefficients = []
        self.thermdispcoef = []
        self._k_wavelength = None
        self._k = None
        self._n_formula = None
        self._n_wavelength = None
        self._n = None
        self._t0 = None
        self.reference_data = None
        self.reference_air_model = "kohlrausch"

        self.formula_map = {
            "formula 1": self._formula_1,
            "formula 2": self._formula_2,
            "formula 3": self._formula_3,
            "formula 4": self._formula_4,
            "formula 5": self._formula_5,
            "formula 6": self._formula_6,
            "formula 7": self._formula_7,
            "formula 8": self._formula_8,
            "formula 9": self._formula_9,
            "tabulated n": self._tabulated_n,
            "tabulated nk": self._tabulated_n,
        }

        data = self._read_file()
        self._parse_file(data)

    def _calculate_absolute_n(self, wavelength, **kwargs):
        """
        Calculates the absolute refractive index from file data, including
        thermal corrections.
        """
        catalog_n = self.formula_map[self._n_formula](wavelength)

        if self.is_relative_to_air:
            # Define the standard conditions for the catalog's reference air
            reference_temp_c = self._t0 or 20.0
            # Pressure is always 101325 Pa (1 atm) for reference
            reference_conditions = EnvironmentalConditions(
                temperature=reference_temp_c, pressure=101325.0, relative_humidity=0.0
            )

            n_air_reference = refractive_index_air(
                wavelength, reference_conditions, model=self.reference_air_model
            )
            absolute_n_reference = catalog_n * n_air_reference
        else:
            absolute_n_reference = catalog_n

        temperature = kwargs.get("temperature")
        if (
            temperature is not None
            and self._t0 is not None
            and self.thermdispcoef
            and be.any(be.array(self.thermdispcoef))
        ):
            c = self.thermdispcoef
            delta_t = temperature - self._t0
            term1 = c[0] + c[1] * delta_t + c[2] * delta_t**2
            term2 = (c[3] + c[4] * delta_t) / (wavelength**2 - c[5] ** 2)
            dn_abs = (
                (absolute_n_reference**2 - 1.0)
                / (2.0 * absolute_n_reference)
                * (term1 + term2)
                * delta_t
            )
            return absolute_n_reference + dn_abs

        return absolute_n_reference

    def _calculate_k(self, wavelength, **kwargs):
        """Retrieves the extinction coefficient of the material at a
        given wavelength. If no exxtinction coefficient data is found, it is
        assumed to be 0 and prints a warning message, only once.
        """
        if self._k is None or self._k_wavelength is None:
            if not self._k_warning_printed:
                material_name = os.path.basename(self.filename)
                print(
                    f"WARNING: No extinction coefficient data found "
                    f"for {material_name}. Assuming it is 0.",
                )
                self._k_warning_printed = True

            if be.is_array_like(wavelength):
                return be.zeros_like(wavelength)
            return 0.0

        return be.interp(wavelength, self._k_wavelength, self._k)

    def _formula_1(self, w):
        """Sellmeier formula"""
        c = self.coefficients
        try:
            n = 1 + c[0]
            for k in range(1, len(c), 2):
                n = n + c[k] * w**2 / (w**2 - c[k + 1] ** 2)
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 1.") from err
        return be.sqrt(n)

    def _formula_2(self, w):
        """Sellmeier-2 formula"""
        c = self.coefficients
        try:
            n = 1 + c[0]
            for k in range(1, len(c), 2):
                n = n + c[k] * w**2 / (w**2 - c[k + 1])
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 2.") from err
        return be.sqrt(n)

    def _formula_3(self, w):
        """Polynomial formula"""
        c = self.coefficients
        try:
            n = c[0]
            for k in range(1, len(c), 2):
                n = n + c[k] * w ** c[k + 1]
            return be.sqrt(n)
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 3.") from err

    def _formula_4(self, w):
        """RefractiveIndex.INFO formula"""
        c = self.coefficients
        try:
            n = (
                c[0]
                + c[1] * w ** c[2] / (w**2 - c[3] ** c[4])
                + c[5] * w ** c[6] / (w**2 - c[7] ** c[8])
            )
            for k in range(9, len(c), 2):
                n = n + c[k] * w ** c[k + 1]
            return be.sqrt(n)
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 4.") from err

    def _formula_5(self, w):
        """Cauchy formula"""
        c = self.coefficients
        try:
            n = c[0]
            for k in range(1, len(c), 2):
                n = n + c[k] * w ** c[k + 1]
            return n
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 5.") from err

    def _formula_6(self, w):
        """Gases formula"""
        c = self.coefficients
        try:
            n = 1 + c[0]
            for k in range(1, len(c), 2):
                n = n + c[k] / (c[k + 1] - w**-2)
            return n
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 6.") from err

    def _formula_7(self, w):
        """Herzberger formula"""
        c = self.coefficients
        try:
            n = c[0] + c[1] / (w**2 - 0.028) + c[2] * (1 / (w**2 - 0.028)) ** 2
            for k in range(3, len(c)):
                n = n + c[k] * w ** (2 * (k - 2))
            return n
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 7.") from err

    def _formula_8(self, w):
        """Retro formula"""
        c = self.coefficients
        if len(c) != 4:
            raise ValueError("Invalid coefficients for dispersion formula 8.")

        b = c[0] + c[1] * w**2 / (w**2 - c[2]) + c[3] * w**2
        return be.sqrt((1 + 2 * b) / (1 - b))

    def _formula_9(self, w):
        """Exotic formula"""
        c = self.coefficients
        if len(c) != 6:
            raise ValueError("Invalid coefficients for dispersion formula 9.")

        n = c[0] + c[1] / (w**2 - c[2]) + c[3] * (w - c[4]) / ((w - c[4]) ** 2 + c[5])
        return be.sqrt(n)

    def _tabulated_n(self, w):
        """Calculate the refractive index using tabulated data."""
        try:
            return be.interp(w, self._n_wavelength, self._n)
        except ValueError as err:
            raise ValueError(
                "No tabular refractive index data found or data is invalid."
            ) from err

    def _read_file(self) -> dict:
        """Read the material YAML file."""
        with open(self.filename) as stream:
            return yaml.safe_load(stream)

    def _set_formula_type(self, formula_type):
        """Set the refractive index formula type."""
        if self._n_formula is None:
            self._n_formula = formula_type
        else:
            raise ValueError("Multiple refractive index formulas found.")

    def _parse_file(self, data: dict) -> None:
        """Parse a material file's structured data."""
        for sub_data in data.get("DATA", []):
            self._parse_sub_data(sub_data)

        self._parse_thermal_dispersion(data)
        self._parse_reference(data)

    def _parse_sub_data(self, sub_data: dict) -> None:
        """Parse a single DATA block from the material file."""
        sub_data_type = sub_data.get("type", "")

        if sub_data_type.startswith("formula "):
            self._parse_formula_data(sub_data, sub_data_type)
        elif sub_data_type.startswith("tabulated"):
            self._parse_tabulated_data(sub_data, sub_data_type)

    def _parse_formula_data(self, sub_data: dict, sub_data_type: str) -> None:
        """Parse formula-based material data."""
        self.coefficients = be.array(
            [float(k) for k in sub_data.get("coefficients", "").split()]
        )
        self.coefficients = be.reshape(self.coefficients, (-1, 1))
        self._set_formula_type(sub_data_type)

    def _parse_tabulated_data(self, sub_data: dict, sub_data_type: str) -> None:
        """Parse tabulated material data."""
        data_file = StringIO(sub_data.get("data", ""))
        numpy_arr = np.loadtxt(data_file)
        arr = be.asarray(numpy_arr)

        if arr.ndim == 1:
            arr = be.reshape(arr, (1, -1) if arr.shape[0] > 0 else (0, 0))

        if sub_data_type == "tabulated n":
            self._n_wavelength, self._n = arr[:, 0], arr[:, 1]
            self._set_formula_type(sub_data_type)
        elif sub_data_type == "tabulated k":
            self._k_wavelength, self._k = arr[:, 0], arr[:, 1]
        elif sub_data_type == "tabulated nk":
            self._n_wavelength = self._k_wavelength = arr[:, 0]
            self._n, self._k = arr[:, 1], arr[:, 2]
            self._set_formula_type(sub_data_type)

    def _parse_thermal_dispersion(self, data: dict) -> None:
        """Parse thermal dispersion and reference temperature data."""
        try:
            coeff = data["SPECS"]["thermal_dispersion"][0]
            if coeff.get("type", "").startswith("Schott"):
                self.thermdispcoef = be.array(
                    [float(k) for k in coeff.get("coefficients", "").split()]
                )
                self.thermdispcoef = be.reshape(self.thermdispcoef, (-1, 1))

            self._t0 = float(data["SPECS"]["temperature"].split(" ")[0])
        except KeyError:
            pass

    def _parse_reference(self, data: dict) -> None:
        """Parse optional reference information."""
        with contextlib.suppress(KeyError):
            self.reference_data = data["REFERENCE"]

    def to_dict(self):
        """Returns the material data as a dictionary."""
        material_dict = super().to_dict()
        material_dict.update(
            {
                "filename": self.filename,
                "is_relative_to_air": self.is_relative_to_air,
            },
        )
        return material_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a material from a dictionary representation."""
        if "filename" not in data:
            raise ValueError("Material file data missing filename.")

        is_relative = data.get("is_relative_to_air", True)
        material = cls(data["filename"], is_relative_to_air=is_relative)
        return material
