"""Multi-Configuration Module

This module provides the MultiConfiguration class for managing optical systems
with multiple configurations, such as zoom lenses.

Kramer Harrison, 2025
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt

from optiland.utils import set_attr_by_path
from optiland.visualization import OpticViewer
from optiland.visualization.themes import get_active_theme

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
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

    def set_property(
        self,
        value: Any,
        configurations: list[int] | Literal["all"] = "all",
        surface_index: int | None = None,
        attribute_path: str = None,
    ):
        """Set a property value across one or more configurations.

        Args:
            value: The value to set.
            configurations: A list of configuration indices to update, or "all"
                to update the base configuration and ensure links (pickups)
                exist (or are created) for other configurations.
            surface_index: The index of the surface to modify. If None, the
                property is assumed to be on the Optic itself.
            attribute_path: The dot-separated path to the attribute, or a
                known alias ('radius', 'thickness', 'conic', 'material').
        """
        if configurations == "all":
            configs_to_update = list(range(len(self.configurations)))
        else:
            configs_to_update = configurations

        # Standardize aliases
        if attribute_path == "radius":
            self.set_radius(surface_index, value, configurations)
            return
        elif attribute_path == "thickness":
            self.set_thickness(surface_index, value, configurations)
            return
        elif attribute_path == "conic":
            self.set_conic(surface_index, value, configurations)
            return
        elif attribute_path == "material":
            self.set_material(surface_index, value, configurations)
            return

        # Generic handling
        for config_idx in configs_to_update:
            if config_idx == 0:
                # Set on base optic
                self._set_generic_value(0, surface_index, attribute_path, value)
            else:
                if configurations == "all":
                    # If setting "all", we want to ensure it is linked to base
                    self._ensure_generic_pickup(
                        config_idx, 0, surface_index, attribute_path
                    )
                else:
                    # If setting specific config (not 0), assume unique val & break link
                    self._remove_generic_pickup(
                        config_idx, surface_index, attribute_path
                    )
                    self._set_generic_value(
                        config_idx, surface_index, attribute_path, value
                    )

    def set_radius(
        self,
        surface_index: int,
        value: float,
        configurations: list[int] | Literal["all"] = "all",
    ):
        """Set the radius of a surface."""
        self._set_standard_property("radius", surface_index, value, configurations)

    def set_thickness(
        self,
        surface_index: int,
        value: float,
        configurations: list[int] | Literal["all"] = "all",
    ):
        """Set the thickness of a surface."""
        self._set_standard_property("thickness", surface_index, value, configurations)

    def set_conic(
        self,
        surface_index: int,
        value: float,
        configurations: list[int] | Literal["all"] = "all",
    ):
        """Set the conic constant of a surface."""
        self._set_standard_property("conic", surface_index, value, configurations)

    def set_material(
        self,
        surface_index: int,
        value: str | BaseMaterial,
        configurations: list[int] | Literal["all"] = "all",
    ):
        """Set the material of a surface."""
        self._set_standard_property("material", surface_index, value, configurations)

    def set_surface_property(
        self,
        surface_index: int,
        attribute_path: str,
        value: Any,
        configurations: list[int] | Literal["all"] = "all",
    ):
        """Convenience wrapper for set_property on a surface."""
        self.set_property(value, configurations, surface_index, attribute_path)

    def set_optic_property(
        self,
        attribute_path: str,
        value: Any,
        configurations: list[int] | Literal["all"] = "all",
    ):
        """Convenience wrapper for set_property on the optic."""
        self.set_property(value, configurations, None, attribute_path)

    def _set_standard_property(self, attr_type, surface_index, value, configurations):
        """Internal helper for standard properties (radius, conic, etc)."""
        if configurations == "all":
            configs_to_update = list(range(len(self.configurations)))
        else:
            configs_to_update = configurations

        for config_idx in configs_to_update:
            if config_idx == 0:
                # Update base
                self._apply_standard_value(0, surface_index, attr_type, value)
            else:
                if configurations == "all":
                    # Ensure pickup
                    if attr_type == "material":
                        self._ensure_generic_pickup(
                            config_idx, 0, surface_index, "material_post"
                        )
                    else:
                        self._ensure_pickup(config_idx, surface_index, attr_type)
                else:
                    # Remove pickup and set value
                    if attr_type == "material":
                        self._remove_generic_pickup(
                            config_idx, surface_index, "material_post"
                        )
                    else:
                        self._remove_pickup(config_idx, surface_index, attr_type)

                    self._apply_standard_value(
                        config_idx, surface_index, attr_type, value
                    )

    def _apply_standard_value(self, config_idx, surface_index, attr_type, value):
        optic = self.configurations[config_idx]
        if attr_type == "radius":
            optic.set_radius(value, surface_index)
        elif attr_type == "conic":
            optic.set_conic(value, surface_index)
        elif attr_type == "thickness":
            optic.set_thickness(value, surface_index)
        elif attr_type == "material":
            optic.set_material(value, surface_index)

    def _set_generic_value(self, config_idx, surface_index, path, value):
        optic = self.configurations[config_idx]
        if surface_index is not None:
            # Relative to surface
            full_path = f"surface_group.surfaces[{surface_index}].{path}"
        else:
            # Relative to optic
            full_path = path

        set_attr_by_path(optic, full_path, value)

    def _ensure_pickup(self, config_idx, surface_index, attr_type):
        """Ensure a standard pickup exists linking to config 0."""
        optic = self.configurations[config_idx]
        # Check if pickup exists
        for p in optic.pickups.pickups:
            if (
                p.target_surface_idx == surface_index
                and p.attr_type == attr_type
                and p.source_optic == self.configurations[0]
            ):
                return  # Exists

        # Create pickup
        optic.pickups.add(
            source_surface_idx=surface_index,
            attr_type=attr_type,
            target_surface_idx=surface_index,
            source_optic=self.configurations[0],
        )

    def _remove_pickup(self, config_idx, surface_index, attr_type):
        """Remove a standard pickup."""
        optic = self.configurations[config_idx]
        to_remove = []
        for p in optic.pickups.pickups:
            if p.target_surface_idx == surface_index and p.attr_type == attr_type:
                to_remove.append(p)

        for p in to_remove:
            optic.pickups.pickups.remove(p)

    def _ensure_generic_pickup(self, config_idx, source_idx, surface_index, path):
        """Ensure a generic pickup exists.

        For Generic Pickups:
        - target_surface_idx: surface_index
        - attr_type: path (e.g. 'geometry.coefficients[0]')
        """
        optic = self.configurations[config_idx]
        source_optic = self.configurations[source_idx]

        if surface_index is not None:
            full_path = f"surface_group.surfaces[{surface_index}].{path}"
        else:
            full_path = path

        # Check existence
        for p in optic.pickups.pickups:
            if p.attr_type == full_path and p.source_optic == source_optic:
                return

        # Add generic pickup
        optic.pickups.add(
            source_surface_idx=0,  # Ignored for generic
            attr_type=full_path,
            target_surface_idx=0,  # Ignored for generic
            source_optic=source_optic,
        )

    def _remove_generic_pickup(self, config_idx, surface_index, path):
        optic = self.configurations[config_idx]
        if surface_index is not None:
            full_path = f"surface_group.surfaces[{surface_index}].{path}"
        else:
            full_path = path

        to_remove = []
        for p in optic.pickups.pickups:
            if p.attr_type == full_path:
                to_remove.append(p)

        for p in to_remove:
            optic.pickups.pickups.remove(p)

    def current_config(self, index: int) -> Optic:
        """Returns the configuration at the given index."""
        return self.configurations[index]

    def draw(
        self,
        figsize: tuple[float, float] | None = None,
        sharex: bool = True,
        sharey: bool = True,
        **kwargs,
    ):
        """Draw the multi-configuration system.

        Args:
            figsize: The size of the figure for a SINGLE configuration.
                The total figure height will be scaled by the number of configs.
                If None, uses the active theme's default figsize.
            sharex: If True, share the x-axis limits and labels.
            sharey: If True, share the y-axis limits and labels.
            **kwargs: Additional arguments passed to OpticViewer.view().
        """
        theme = get_active_theme()
        params = theme.parameters

        if figsize is None:
            figsize = params["figure.figsize"]

        num_configs = len(self.configurations)
        total_height = figsize[1] * num_configs
        fig, axes = plt.subplots(
            nrows=num_configs,
            figsize=(figsize[0], total_height),
            sharex=sharex,
            sharey=sharey,
        )
        fig.set_facecolor(params["figure.facecolor"])

        # Ensure axes is iterable
        if num_configs == 1:
            axes = [axes]
        elif hasattr(axes, "flat"):
            axes = axes.flatten()

        for i, (optic, ax) in enumerate(zip(self.configurations, axes, strict=False)):
            ax.set_facecolor(params["axes.facecolor"])

            viewer = OpticViewer(optic)

            # Handle title (append config name if title exists)
            plot_kwargs = kwargs.copy()
            base_title = plot_kwargs.get("title")

            if base_title:
                plot_kwargs["title"] = f"{base_title} (Config {i})"
            else:
                plot_kwargs["title"] = f"Configuration {i}"

            viewer.view(ax=ax, **plot_kwargs)

        plt.tight_layout()
        return fig, axes
