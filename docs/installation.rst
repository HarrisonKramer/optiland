Installation
===================================

.. _install:

Optiland can be installed via pip:

.. code-block:: console

   > pip install optiland

You can now run your first ratrace and visualize the system:

.. code:: python

   from optiland.samples.objectives import ReverseTelephoto

   lens = ReverseTelephoto()
   lens.draw3D()
