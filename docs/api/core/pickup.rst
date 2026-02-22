pickup
======

.. automodule:: pickup

   
   .. rubric:: Classes

   .. autosummary::
   
      Pickup
      PickupManager

   .. rubric:: Notes
   
   When defining generic paths for attributes on surfaces using ``PickupManager.add`` or ``Pickup`` directly, you can now use ``[i]`` to represent the surface index. For instance, to pickup aspheric coefficients, you can use ``attr_type="surface_group.surfaces[i].geometry.coefficients"``. The ``[i]`` tag will automatically be resolved to both the target and source index when extracting and setting values.
   