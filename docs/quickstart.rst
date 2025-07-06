Quickstart
==========

.. _first_script:

Once you have installed Optiland, you can start designing and analyzing optical systems. Here is a simple example which loads and visualizes a Cooke Triplet lens system.

Optiland "Hello, World"
-----------------------

.. code-block:: python

   from optiland.samples.objectives import CookeTriplet

   lens = CookeTriplet()
   lens.draw3D()

.. figure:: images/cooke.png
   :alt: Cooke Triplet Lens System
   :align: center

   This shows the resulting 3D visualization of the Cooke triplet lens system.

Running the GUI
---------------

Optiland also includes a Graphical User Interface (GUI) for interactive design and analysis. To run the GUI, you can typically use the following command in your terminal (make sure your environment with Optiland installed is active):

.. code-block:: bash

   python -m optiland_gui.run_gui

This will launch the main GUI window, allowing you to create, modify, and analyze optical systems interactively.

Optiland for Beginners
----------------------

This script is the first of the learning guide series. It introduces the basic concepts of Optiland and demonstrates how to create a simple lens system.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Optiland for Beginners <examples/Tutorial_1a_Optiland_for_Beginners>
