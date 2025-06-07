Visualization
=============

This section contains the visualization modules of Optiland, which enable 2D and 3D visualization of optical systems.

Visualization is performed in a modular manner to allow for easy customization and extension. Separate
modules for visualization of surfaces, lenses, and rays are provided, both for 2D and 3D. Visualization
of the full system, including rays, is orchestrated by the `optiland.visualization.OpticViewer` or
`optiland.visualization.OpticViewer3D` classes.

.. autosummary::
   :toctree: visualization/
   :caption: Legacy Visualization Modules

   visualization.lens
   visualization.mirror
   visualization.rays
   visualization.surface
   visualization.system
   visualization.utils
   visualization.visualization

Generic Plotting Utilities
--------------------------

Optiland also provides a general-purpose plotting utility class, `optiland.plotting.Plotter`,
which is the primary interface for creating various 2D and 3D plots such as line charts,
scatter plots, image plots, and more. This system is used by many analysis modules.

Key features of the `optiland.plotting` sub-package include:

*   **Versatile Plotting Methods:** Static methods on `Plotter` for common plot types.
*   **Theming:** Consistent visual styles via `optiland.plotting.themes`.
*   **Global Configuration:** System-wide plot settings via `optiland.plotting.config`.
*   **Legend Customization:** Fine-grained control over legends using `optiland.plotting.plot_configs.LegendConfig`.

For detailed API information, please refer to the docstrings of the respective modules and classes.

.. rubric:: Plotting Core API

.. automodule:: optiland.plotting.core
   :members: Plotter

.. rubric:: Plotting Configuration API

.. automodule:: optiland.plotting.plot_configs
   :members: LegendConfig
