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

.. image:: ../images/telephoto.png
   :align: center

|br|

Python code to generate this 3D visualization:

.. code:: python

   from optiland.samples.objectives import ReverseTelephoto

   lens = ReverseTelephoto()
   lens.draw3D()


Getting Started
===============

Installation
------------

.. _install:

.. toctree::
   :maxdepth: 2

   installation
   first_steps


Core Functionalities
====================

.. _functionalities:

.. toctree::
   :maxdepth: 2

* Lens entry
* 2D/3D visualization
* Paraxial and aberration analyses
* Real and paraxial ray tracing, including aspherics and freeforms
* Polarization ray tracing
* Real analysis functions (spot diagrams, ray aberration fans, OPD fans, distortion, PSF, MTF, etc.)
* Glass and material catalogue (based on refractiveindex.info)
* Design optimization (local and global)
* Wavefront and Zernike analysis
* Tolerancing, including sensitivity analysis and Monte Carlo methods
* Coating and surface scatter (BSDF) analysis
* Zemax file import


.. |br| raw:: html

      <br>