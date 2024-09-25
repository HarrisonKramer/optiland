Welcome to Optiland's documentation!
====================================

.. note::

   This project is under active development.

**Optiland** is a powerful, Python-based, open-source lens design and analysis framework.
With its intuitive interface, Optiland enables the design, optimization, and analysis of complex
optical systems, from paraxial and real raytracing to advanced polarization, coatings, and
wavefront analyses. It supports 2D/3D visualization, comprehensive tolerancing, global
optimization, and freeform optics, among other features. Built on the speed of NumPy and
SciPy, Optiland ensures high-performance handling of intricate optical computations, delivering
professional-grade results in an open, flexible environment.

.. image:: images/telephoto.png
   :align: center

|br|

Python code to generate this 3D visualization:

.. code:: python

   from optiland.samples.objectives import ReverseTelephoto

   lens = ReverseTelephoto()
   lens.draw3D()


Getting Started
===============

.. _getting_started:

.. toctree::
   :maxdepth: 1

   installation
   quickstart


Core Functionalities
====================

.. toctree::
   :maxdepth: 2

   functionalities


Example Gallery
===============

.. toctree::
   :maxdepth: 1
   :titlesonly:

   example_gallery


Learning Guide
==============

.. toctree::
   :maxdepth: 1

   learning_guide


Contributing
============

.. toctree::
   :maxdepth: 1

   contributing


Authors
=======

.. toctree::
   :maxdepth: 1

   authors


License
=======

.. toctree::
   :maxdepth: 1

   license


.. |br| raw:: html

      <br>
