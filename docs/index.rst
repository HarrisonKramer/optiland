Welcome to Optiland's documentation!
====================================

.. note::

   This project is under active development.

**Optiland** is a Python-based, open-source lens design and analysis framework.
With a simple and intuitive Python interface, Optiland enables the design, optimization, and analysis of complex
optical systems, from paraxial and real raytracing to polarization, coatings, and
wavefront analyses. It supports 2D/3D visualization, comprehensive tolerancing, local and global
optimization, and freeform optics, among other features. Built on the speed of NumPy and SciPy,
Optiland delivers computational efficiency and flexibility across a wide range of optical tasks.

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

   gallery/introduction
   gallery/basic_lenses
   gallery/specialized_lenses
   gallery/reflective_systems
   gallery/analysis
   gallery/opd_psf_mtf
   gallery/optimization
   gallery/tolerancing
   gallery/freeforms
   gallery/miscellaneous


.. _learning_guide:

.. toctree::
   :maxdepth: 1
   :caption: Learning Guide

   learning_guide


.. toctree::
   :maxdepth: 1
   :caption: Developer's Guide
   :numbered:

   developers_guide/introduction
   developers_guide/requirements
   developers_guide/installation
   developers_guide/getting_started
   developers_guide/architecture
   developers_guide/ray_overview
   developers_guide/surface_overview
   developers_guide/geometry_overview
   developers_guide/analysis_framework
   developers_guide/optimization_framework
   developers_guide/tolerancing_framework
   developers_guide/visualization_framework
   developers_guide/optiland_file_format


.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/api_introduction
   api/api_analysis
   api/api_coatings
   api/api_core
   api/api_fileio
   api/api_geometries
   api/api_materials
   api/api_wavefront
   api/api_optic
   api/api_optimization
   api/api_paraxial
   api/api_physical_apertures
   api/api_rays
   api/api_raytrace
   api/api_surfaces
   api/api_tolerancing
   api/api_visualization
   api/api_zernike


.. toctree::
   :maxdepth: 1
   :caption: Authors

   authors


.. toctree::
   :maxdepth: 1
   :caption: License

   license

.. toctree::
   :maxdepth: 1
   :caption: References

   references

.. |br| raw:: html

      <br>
