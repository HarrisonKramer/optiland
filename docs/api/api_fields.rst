Fields
=======

In optical design, a field defines the angular or spatial extent from which light rays enter the optical system. Fields determine the system's field of view and are essential for ray tracing and performance evaluation.

Optiland supports three field specification types:
* Angle (`angle`): Field positions specified as angular coordinates
* Object height (`object_height`): Field positions defined by object space heights
* Paraxial image height (`paraxial_image_height`): Field positions specified by paraxial image plane heights

This section covers Optiland functionality related to defining and manipulating various field types used in optical systems.

.. autosummary::
   :toctree: fields/
   :caption: Field Modules

   fields.field_group
   fields.field_types
   fields.field
