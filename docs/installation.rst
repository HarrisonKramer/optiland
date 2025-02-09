.. _installation:

Installation
============

Optiland can be installed via pip or directly from source.

Installing via pip
------------------

To install Optiland via pip, run the following command in your terminal:

.. code-block:: console

   pip install optiland

Installing from Source
----------------------

To install Optiland from source, follow these steps:

1. Clone the repository from GitHub:

   .. code-block:: console

      git clone https://github.com/HarrisonKramer/optiland.git
      cd optiland

2. Install Optiland and its dependencies:

   .. code-block:: console

      pip install .

Verify Installation
-------------------

You can verify installation by importing Optiland in Python.

.. code-block:: python

   import optiland

Optionally, you may generate and visualize a lens system:

.. code-block:: python

   from optiland.samples.objectives import ReverseTelephoto

   lens = ReverseTelephoto()
   lens.draw3D()


Note that import may take longer the first time Optiland is run due to caching and compilation of code. Subsequent imports will be faster.
