"""System Visualization Module

This module contains the OpticalSystem class for visualizing optical systems.

Kramer Harrison, 2024
"""

from optiland.visualization.lens import Lens2D, Lens3D
from optiland.visualization.mirror import Mirror3D
from optiland.visualization.surface import Surface2D, Surface3D


class OpticalSystem:
    """A class to represent an optical system for visualization. The optical
    system contains surfaces and lenses.

    Args:
        optic (Optic): The optical system to be used for plotting.
        rays (Rays): The rays interacting with the optical system.
        projection (str): The type of projection for visualization.
            Must be '2d' or '3d'.

    Attributes:
        optic (Optic): The optical system to be used for plotting.
        rays (Rays): The rays interacting with the optical system.
        projection (str): The type of projection for visualization.
            Must be '2d' or '3d'.
        components (list): A list to store the components of the optical
            system.
        component_registry (dict): A registry mapping component names to their
            respective classes for 2D and 3D projections.

    Methods:
        plot(ax):
            Identifies and plots the components of the optical system on the
                given axis (or renderer for 3D plotting).

    """

    def __init__(self, optic, rays, projection="2d"):
        self.optic = optic
        self.rays = rays
        self.projection = projection
        self.components = []  # initialize empty list of components

        if self.projection not in ["2d", "3d"]:
            raise ValueError("Invalid projection type. Must be '2d' or '3d'.")

        self.component_registry = {
            "lens": {"2d": Lens2D, "3d": Lens3D},
            "mirror": {"2d": Surface2D, "3d": Mirror3D},
            "surface": {"2d": Surface2D, "3d": Surface3D},
        }

    def plot(self, ax):
        """Plots the components of the optical system on the given
        axis (or renderer for 3D plotting).
        """
        self._identify_components()
        for component in self.components:
            component.plot(ax)

    def _identify_components(self):
        """Identifies the components of the optical system and adds them to the
        list of components.
        """
        self.components = []
        n = self.optic.n()  # refractive indices
        num_surf = self.optic.surface_group.num_surfaces

        lens_surfaces = []

        for k, surf in enumerate(self.optic.surface_group.surfaces):
            # Get the surface extent
            extent = self.rays.r_extent[k]

            # Object surface
            if k == 0:
                if not surf.is_infinite:
                    self._add_component("surface", surf, extent)

            # Image surface or paraxial surface
            elif k == num_surf - 1 or surf.surface_type == "paraxial":
                self._add_component("surface", surf, extent)

            # Surface is a mirror
            elif surf.is_reflective:
                self._add_component("mirror", surf, extent)

            # Front surface of a lens
            elif n[k] > 1:
                surface = self._get_lens_surface(surf, extent)
                lens_surfaces.append(surface)

            # Back surface of a lens
            elif n[k] == 1 and n[k - 1] > 1 and lens_surfaces:
                surface = self._get_lens_surface(surf, extent)
                lens_surfaces.append(surface)
                self._add_component("lens", lens_surfaces)

                lens_surfaces = []

            # add final lens, if any
            if lens_surfaces:
                self._add_component("lens", lens_surfaces)

    def _add_component(self, component_name, *args):
        """Adds a component to the list of components."""
        if component_name in self.component_registry:
            component_class = self.component_registry[component_name][self.projection]
        else:
            raise ValueError(f"Component {component_name} not found in registry.")

        self.components.append(component_class(*args))

    def _get_lens_surface(self, surface, *args):
        """Gets the lens surface based on the projection type."""
        surface_class = self.component_registry["surface"][self.projection]
        return surface_class(surface, *args)
