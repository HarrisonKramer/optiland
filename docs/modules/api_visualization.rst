Visualization
=============

This section contains the visualization modules of Optiland, which enable 2D and 3D visualization of optical systems.

Visualization is performed in a modular manner to allow for easy customization and extension. Separate
modules for visualization of surfaces, lenses, and rays are provided, both for 2D and 3D. Visualization
of the full system, including rays, is orchestrated by the `optiland.visualizations.OpticViewer` or
`optiland.visualizations.OpticViewer3D` classes.

.. autosummary::
   :toctree: visualization/
   :caption: visualization Modules

   optiland.visualization.lens
   optiland.visualization.mirror
   optiland.visualization.rays
   optiland.visualization.surface
   optiland.visualization.system
   optiland.visualization.utils
   optiland.visualization.visualization
