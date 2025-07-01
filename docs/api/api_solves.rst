Solves
======

This section includes the various solve classes, which are functions that can be used to set specific values in a system.
For example, a marginal ray height solve can be used to enforce the height of the marginal ray on a specified surface. This
is commonly used to put the image plane at the paraxial image location.

.. autosummary::
   :toctree: solves/
   :caption: Solve Modules

   solves.base
   solves.chief_ray_height
   solves.factory
   solves.marginal_ray_height
   solves.quick_focus
   solves.ray_height_base
   solves.solve_manager
