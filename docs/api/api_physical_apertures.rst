Physical Apertures
==================

This section provides an overview of the physical apertures available in the
Optiland package. Physical apertures are used to define boundaries of the
individual optical surface, i.e., where rays may interact on the surface. For
example, a primary telescope mirror that has a central hole may be defined by a
physical aperture that is circular with a hole in the center (i.e., `RadialAperture`).

.. autosummary::
   :toctree: physical_apertures/
   :caption: Physical Aperture Modules

   physical_apertures.base
   physical_apertures.radial
   physical_apertures.rectangular
   physical_apertures.elliptical
   physical_apertures.offset_radial
   physical_apertures.polygon
   physical_apertures.file
