"""Material File

This module contains a class for representing a material based on a material
YAML file from the refractiveindex.info database.

Kramer Harrison, 2024
"""

from __future__ import annotations

import contextlib
import os
from io import StringIO
from typing import Any, Callable

import numpy as np
import yaml

import optiland.backend as be
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
            relative to air at standard conditions. If False, the refractive
            index is calculated relative to vacuum. Defaults to True.

    Attributes:
        filename (str): The path to the material's YAML file.
        is_relative_to_air (bool): Flag indicating if the index is relative to air.
        coefficients (be.ndarray | list): Coefficients for dispersion formulas.
        thermdispcoef (be.ndarray | list): Coefficients for thermal dispersion.
        reference_data (dict | None): Optional reference information from the file.
        formula_map (dict[str, Callable]): Maps formula names to methods.
    """

    def __init__(self, filename: str, is_relative_to_air: bool = True):
        super().__init__()
        self.filename = filename
        self.is_relative_to_air = is_relative_to_air
        self._k_warning_printed = False

        # Initialize attributes with type hints
        self.coefficients: be.ndarray | list = []
        self.thermdispcoef: be.ndarray | list = []
        self._k_wavelength: be.ndarray | None = None
        self._k: be.ndarray | None = None
        self._n_formula: str | None = None
        self._n_wavelength: be.ndarray | None = None
        self._n: be.ndarray | None = None
        self._t0: float | None = None
        self.reference_data: dict[str, Any] | None = None
        self.reference_air_model = "kohlrausch"

        self.formula_map: dict[str, Callable] = {
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

    def _calculate_absolute_n(
        self, wavelength: float | be.ndarray, **kwargs: Any
    ) -> float | be.ndarray:
        """
        Calculates the absolute refractive index from file data, including
        thermal corrections.

        Args:
            wavelength (float | be.ndarray): The wavelength(s) in microns.
            **kwargs: Keyword arguments, may include 'temperature'.

        Returns:
            float | be.ndarray: The absolute refractive index.
        """
        from optiland.environment.conditions import EnvironmentalConditions
        from optiland.materials.air import Air

        if self._n_formula is None:
            raise RuntimeError(f"No refractive index formula found for {self.filename}")

        catalog_n = self.formula_map[self._n_formula](wavelength)

        if self.is_relative_to_air:
            # Use a standard air reference, not the global environment
            standard_air = Air(conditions=EnvironmentalConditions())
            n_air_reference = standard_air._calculate_absolute_n(
                wavelength, **kwargs
            )
            absolute_n_reference = catalog_n * n_air_reference
        else:
            absolute_n_reference = catalog_n

        temperature = kwargs.get("temperature")
        if (
            temperature is not None
            and self._t0 is not None
            and self.thermdispcoef
            and be.any(be.asarray(self.thermdispcoef))
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

    def _calculate_k(
        self, wavelength: float | be.ndarray, **kwargs: Any
    ) -> float | be.ndarray:
        """Retrieves the extinction coefficient of the material.

        If no extinction coefficient data is found, it is assumed to be 0.

        Args:
            wavelength (float | be.ndarray): The wavelength(s) in microns.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            float | be.ndarray: The extinction coefficient.
        """
        if self._k is None or self._k_wavelength is None:
            if not self._k_warning_printed:
                material_name = os.path.basename(self.filename)
                print(
                    f"WARNING: No extinction coefficient data found "
                    f"for {material_name}. Assuming it is 0.",
                )
                self._k_warning_printed = True

            if be.is_array_like(wavelength) and be.size(wavelength) > 1:
                return be.zeros_like(wavelength)
            return 0.0

        return be.interp(wavelength, self._k_wavelength, self._k)

    def _formula_1(self, w: float | be.ndarray) -> float | be.ndarray:
        """Sellmeier formula."""
        c = self.coefficients
        try:
            w_sq = be.power(w, 2)
            n_sq = 1.0 + c[0]
            for k in range(1, len(c), 2):
                n_sq = n_sq + c[k] * w_sq / (w_sq - be.power(c[k + 1], 2))
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 1.") from err
        return be.sqrt(n_sq)

    def _formula_2(self, w: float | be.ndarray) -> float | be.ndarray:
        """Sellmeier-2 formula."""
        c = self.coefficients
        try:
            w_sq = be.power(w, 2)
            n_sq = 1.0 + c[0]
            for k in range(1, len(c), 2):
                n_sq = n_sq + c[k] * w_sq / (w_sq - c[k + 1])
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 2.") from err
        return be.sqrt(n_sq)

    def _formula_3(self, w: float | be.ndarray) -> float | be.ndarray:
        """Polynomial formula."""
        c = self.coefficients
        try:
            n_sq = c[0]
            for k in range(1, len(c), 2):
                n_sq = n_sq + c[k] * be.power(w, c[k + 1])
            return be.sqrt(n_sq)
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 3.") from err

    def _formula_4(self, w: float | be.ndarray) -> float | be.ndarray:
        """RefractiveIndex.INFO formula."""
        c = self.coefficients
        try:
            w_sq = be.power(w, 2)
            n_sq = (
                c[0]
                + c[1] * be.power(w, c[2]) / (w_sq - be.power(c[3], c[4]))
                + c[5] * be.power(w, c[6]) / (w_sq - be.power(c[7], c[8]))
            )
            for k in range(9, len(c), 2):
                n_sq = n_sq + c[k] * be.power(w, c[k + 1])
            return be.sqrt(n_sq)
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 4.") from err

    def _formula_5(self, w: float | be.ndarray) -> float | be.ndarray:
        """Cauchy formula."""
        c = self.coefficients
        try:
            n = c[0]
            for k in range(1, len(c), 2):
                n = n + c[k] * be.power(w, c[k + 1])
            return n
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 5.") from err

    def _formula_6(self, w: float | be.ndarray) -> float | be.ndarray:
        """Gases formula."""
        c = self.coefficients
        try:
            n = 1.0 + c[0]
            for k in range(1, len(c), 2):
                n = n + c[k] / (c[k + 1] - be.power(w, -2))
            return n
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 6.") from err

    def _formula_7(self, w: float | be.ndarray) -> float | be.ndarray:
        """Herzberger formula."""
        c = self.coefficients
        try:
            l_sq = be.power(w, 2)
            n = (
                c[0]
                + c[1] * l_sq / (l_sq - 0.028)
                + c[2] * be.power(l_sq / (l_sq - 0.028), 2)
            )
            for k in range(3, len(c)):
                n = n + c[k] * be.power(l_sq, k - 2)
            return n
        except IndexError as err:
            raise ValueError("Invalid coefficients for dispersion formula 7.") from err

    def _formula_8(self, w: float | be.ndarray) -> float | be.ndarray:
        """Retro formula."""
        c = self.coefficients
        if len(c) != 4:
            raise ValueError("Invalid coefficients for dispersion formula 8.")
        l_sq = be.power(w, 2)
        b = c[0] + c[1] * l_sq / (l_sq - c[2]) + c[3] * l_sq
        return be.sqrt((1 + 2 * b) / (1 - b))

    def _formula_9(self, w: float | be.ndarray) -> float | be.ndarray:
        """Exotic formula."""
        c = self.coefficients
        if len(c) != 6:
            raise ValueError("Invalid coefficients for dispersion formula 9.")

        w_sq = be.power(w, 2)
        n_sq = (
            c[0]
            + c[1] / (w_sq - c[2])
            + c[3] * (w - c[4]) / (be.power(w - c[4], 2) + c[5])
        )
        return be.sqrt(n_sq)

    def _tabulated_n(self, w: float | be.ndarray) -> float | be.ndarray:
        """Calculate the refractive index using tabulated data."""
        if self._n is None or self._n_wavelength is None:
            raise ValueError("No tabular refractive index data found.")
        return be.interp(w, self._n_wavelength, self._n)

    def _read_file(self) -> dict[str, Any]:
        """Read the material YAML file."""
        with open(self.filename, encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def _set_formula_type(self, formula_type: str) -> None:
        """Set the refractive index formula type, ensuring only one is set."""
        if self._n_formula is None:
            self._n_formula = formula_type
        else:
            raise ValueError("Multiple refractive index formulas found in file.")

    def _parse_file(self, data: dict[str, Any]) -> None:
        """Parse a material file's structured data."""
        for sub_data in data.get("DATA", []):
            self._parse_sub_data(sub_data)
        self._parse_thermal_dispersion(data)
        self._parse_reference(data)

    def _parse_sub_data(self, sub_data: dict[str, Any]) -> None:
        """Parse a single DATA block from the material file."""
        sub_data_type = sub_data.get("type", "")
        if sub_data_type.startswith("formula "):
            self._parse_formula_data(sub_data, sub_data_type)
        elif sub_data_type.startswith("tabulated"):
            self._parse_tabulated_data(sub_data, sub_data_type)

    def _parse_formula_data(self, sub_data: dict[str, Any], sub_data_type: str) -> None:
        """Parse formula-based material data."""
        self.coefficients = be.asarray(
            [float(k) for k in sub_data.get("coefficients", "").split()]
        )
        self._set_formula_type(sub_data_type)

    def _parse_tabulated_data(
        self, sub_data: dict[str, Any], sub_data_type: str
    ) -> None:
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
            self._n_wavelength, self._k_wavelength = arr[:, 0], arr[:, 0]
            self._n, self._k = arr[:, 1], arr[:, 2]
            self._set_formula_type(sub_data_type)

    def _parse_thermal_dispersion(self, data: dict[str, Any]) -> None:
        """Parse thermal dispersion and reference temperature data."""
        try:
            specs = data["SPECS"]
            if "thermal_dispersion" in specs:
                coeff = specs["thermal_dispersion"][0]
                if coeff.get("type", "").startswith("Schott"):
                    self.thermdispcoef = be.asarray(
                        [float(k) for k in coeff.get("coefficients", "").split()]
                    )
            if "temperature" in specs:
                self._t0 = float(specs["temperature"].split(" ")[0])
        except (KeyError, IndexError):
            pass

    def _parse_reference(self, data: dict[str, Any]) -> None:
        """Parse optional reference information."""
        with contextlib.suppress(KeyError):
            self.reference_data = data["REFERENCE"]

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> MaterialFile:
        """Creates a material from a dictionary representation."""
        if "filename" not in data:
            raise ValueError("Material file data missing filename.")

        is_relative = data.get("is_relative_to_air", True)
        return cls(data["filename"], is_relative_to_air=is_relative)