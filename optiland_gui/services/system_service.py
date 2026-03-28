"""System-level service for the Optiland GUI.

Manages aperture, fields, and wavelength data retrieval for the System
Properties panel.
"""

from __future__ import annotations

import optiland.aperture  # noqa: F401 — ensures all subclasses are registered
import optiland.backend as be  # noqa: F401  (available for future use)
from optiland.aperture import BaseSystemAperture


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

    def get_aperture_types(self) -> list[str]:
        """Return all aperture type keys registered with :class:`BaseSystemAperture`.

        Enumerates the ``_registry`` class attribute that is populated when
        each :class:`~optiland.aperture.base.BaseSystemAperture` subclass is
        imported.

        Returns:
            A sorted list of aperture type identifier strings (e.g.
            ``["EPD", "imageFNO", "objectNA", ...]``).
        """
        return sorted(BaseSystemAperture._registry.keys())

    def get_field_types(self) -> list[tuple[str, str]]:
        """Return all four supported field types.

        Returns:
            A list of ``(display_name, type_key)`` tuples for every field
            type exposed by the GUI:

            - ``("Angle", "angle")``
            - ``("Object Height", "object_height")``
            - ``("Paraxial Image Height", "paraxial_image_height")``
            - ``("Real Image Height", "real_image_height")``
        """
        return [
            ("Angle", "angle"),
            ("Object Height", "object_height"),
            ("Paraxial Image Height", "paraxial_image_height"),
            ("Real Image Height", "real_image_height"),
        ]
