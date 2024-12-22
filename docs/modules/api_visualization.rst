Visualization
=============

This section contains the visualization modules of Optiland, which enable 2D and 3D visualization of optical systems.

Visualization is performed in a modular manner to allow for easy customization and extension. Separate
modules for visualization of surfaces, lenses, and rays are provided, both for 2D and 3D. Visualization
of the full system, including rays, is orchestrated by the `optiland.visualization.OpticViewer` or
`optiland.visualization.OpticViewer3D` classes.

.. autosummary::
   :toctree: visualization/
   :caption: visualization Modules

   visualization.lens
   visualization.mirror
   visualization.rays
   visualization.surface
   visualization.system
   visualization.utils
   visualization.visualization
