"""System-level service for the Optiland GUI.

Manages aperture, fields, and wavelength data retrieval for the System
Properties panel.
"""

from __future__ import annotations

import optiland.backend as be  # noqa: F401  (available for future use)


class SystemService:
    """Provides field and wavelength option lists for the GUI.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`
            instance that owns this service.
    """

    def __init__(self, connector: object) -> None:
        self._connector = connector

    def get_field_options(self) -> list[tuple[str, str]]:
        """Return field selector options for the analysis panel.

        Each option is a ``(display_name, value_str)`` tuple where
        *value_str* is suitable for passing directly to an analysis class.

        Returns:
            A list of tuples starting with ``("all", "'all'")`` followed by
            one entry per field.
        """
        optic = self._connector._optic
        if not optic or optic.fields.num_fields == 0:
            return [("all", "'all'")]
        options = [("all", "'all'")]
        field_coords = optic.fields.get_field_coords()
        for i, field_tuple in enumerate(field_coords):
            hx, hy = field_tuple[0], field_tuple[1]
            display_name = f"Field {i + 1}: ({hx:.3f}, {hy:.3f})"
            value_str = f"[({hx}, {hy})]"
            options.append((display_name, value_str))
        return options

    def get_wavelength_options(self) -> list[tuple[str, str]]:
        """Return wavelength selector options for the analysis panel.

        Each option is a ``(display_name, value_str)`` tuple.

        Returns:
            A list of tuples starting with ``("all", "'all'")`` and
            ``("primary", "'primary'")`` followed by one entry per wavelength.
        """
        optic = self._connector._optic
        if not optic or optic.wavelengths.num_wavelengths == 0:
            return [("all", "'all'"), ("primary", "'primary'")]
        options = [("all", "'all'"), ("primary", "'primary'")]
        wavelength_values = optic.wavelengths.get_wavelengths()
        for wl_value in wavelength_values:
            display_name = f"{wl_value:.4f} µm"
            value_str = f"[{wl_value}]"
            options.append((display_name, value_str))
        return options
