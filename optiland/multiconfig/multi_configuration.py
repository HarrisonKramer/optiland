"""Multi-Configuration Module

This module provides the MultiConfiguration class for managing optical systems
with multiple configurations, such as zoom lenses.

Kramer Harrison, 2025
"""

from __future__ import annotations

import copy  # Use standard copy module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.optic import Optic


class MultiConfiguration:
    """Manages multiple configurations of an optical system.

    This class holds a list of independent Optic instances, one for each
    configuration. It provides methods to create new configurations
    derived from a base configuration and link them using Pickups.

    Args:
        base_optic (Optic): The initial optical system (Configuration 0).

    Attributes:
        configurations (list[Optic]): The list of Optic instances.
    """

    def __init__(self, base_optic: Optic):
        self.configurations: list[Optic] = [base_optic]

    def add_configuration(self, source_config_idx: int = 0) -> Optic:
        """Creates a new configuration based on a source configuration.

        The new configuration is a deep copy of the source. By default,
        Pickups are added to the new configuration that link all its
        surface geometries and basic properties back to the source.
        This ensures that, initially, both configurations are identical
        and controlled by the source's variables.

        Args:
            source_config_idx (int): The index of the configuration to copy.
                Defaults to 0.

        Returns:
            Optic: The new configuration instance.
        """
        source_optic = self.configurations[source_config_idx]
        new_optic = copy.deepcopy(source_optic)
        self.configurations.append(new_optic)

        # Link the new optic to the source optic
        self._link_configurations(source_optic, new_optic)

        return new_optic

    def _link_configurations(self, source: Optic, target: Optic):
        """Internal method to link generic surface properties."""
        # Link Radii and Conics
        for i, (surf_s, _surf_t) in enumerate(
            zip(
                source.surface_group.surfaces,
                target.surface_group.surfaces,
                strict=False,
            )
        ):
            # Radius
            if hasattr(surf_s.geometry, "radius"):
                target.pickups.add(
                    source_surface_idx=i,
                    attr_type="radius",
                    target_surface_idx=i,
                    source_optic=source,
                )

            # Conic
            if hasattr(surf_s.geometry, "k"):
                target.pickups.add(
                    source_surface_idx=i,
                    attr_type="conic",
                    target_surface_idx=i,
                    source_optic=source,
                )

            # Thickness (except last surface)
            if i < len(source.surface_group.surfaces) - 1:
                target.pickups.add(
                    source_surface_idx=i,
                    attr_type="thickness",
                    target_surface_idx=i,
                    source_optic=source,
                )

    def current_config(self, index: int) -> Optic:
        """Returns the configuration at the given index."""
        return self.configurations[index]
