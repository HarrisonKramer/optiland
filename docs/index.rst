Welcome to Optiland's documentation!
====================================

.. note::

   This project is under active development.

**Optiland** is a Python-based, open-source lens design and analysis framework.
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


.. _getting_started:

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart


.. toctree::
   :maxdepth: 2
   :caption: Core Functionalities

   functionalities


.. toctree::
   :maxdepth: 1
   :caption: Example Gallery
   :titlesonly:

   example_gallery


.. toctree::
   :maxdepth: 1
   :caption: Learning Guide

   learning_guide


.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide



.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   modules/modules


.. toctree::
   :maxdepth: 1
   :caption: Authors

   authors


.. toctree::
   :maxdepth: 1
   :caption: License

   license


.. |br| raw:: html

      <br>
