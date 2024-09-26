Welcome to Optiland's documentation!
====================================

.. note::

   This project is under active development.

**Optiland** is a powerful, Python-based, open-source lens design and analysis framework.
With a simple and intuitive Python interface, Optiland enables the design, optimization, and analysis of complex
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
   :caption: Getting Started

   installation
   quickstart


Core Functionalities
====================

.. toctree::
   :maxdepth: 2
   :caption: Core Functionalities

   functionalities


Example Gallery
===============

.. toctree::
   :maxdepth: 1
   :caption: Example Gallery
   :titlesonly:

   example_gallery


Learning Guide
==============

.. toctree::
   :maxdepth: 1
   :caption: Learning Guide

   learning_guide


Developer Guide
===============

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide


Contributing
============

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing


API Reference
=============

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   modules/modules


Authors
=======

.. toctree::
   :maxdepth: 1
   :caption: Authors

   authors


License
=======

.. toctree::
   :maxdepth: 1
   :caption: License

   license


.. |br| raw:: html

      <br>
