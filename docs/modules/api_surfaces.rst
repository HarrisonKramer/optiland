Surfaces
========

This section describes the surface modules in Optiland. Surfaces are the building blocks of optical systems, 
and Optiland provides a variety of surface types and properties to support the design and analysis of complex optical systems.

Optiland builds individual surfaces into the so-called `SurfaceGroup`, which represents a collection of surfaces that form an optical system.
Most analyses are performed on the `SurfaceGroup` level, which abstracts the complexity of individual surfaces and provides a unified interface for optical system design and analysis.

.. autosummary::
   :toctree: surfaces/
   :caption: Surface Modules

   surfaces.image_surface
   surfaces.object_surface
   surfaces.standard_surface
   surfaces.surface_factory
   surfaces.surface_group
