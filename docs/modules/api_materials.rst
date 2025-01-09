Materials
=========

This section provides a comprehensive overview of the materials available in Optiland.
A material is a set of properties that define the optical behavior of a medium. Specifically in Optiland, a material defines the refractive
index and extinction coefficient of a medium at any wavelength. A material may be described simply by refractive index and abbe number, or it may be a more complex
model based on Sellmeier, or other, coefficients. Optiland provides a wide range of materials, which are outlined here.

Optiland includes a database of common materials based on refrativeindex.info. Any material in this database
can be accessed by name via the Material class.

.. autosummary::
   :toctree: materials/
   :caption: Material Modules

   materials.abbe
   materials.ideal
   materials.material_file
   materials.material
