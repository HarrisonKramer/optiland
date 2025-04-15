Optimization
============

This section covers the optimization functionality of the Optiland package. The optimization module
provides a set of tools for optimizing optical systems. It includes a variety of optimization algorithms,
such as gradient-based and evolutionary algorithms, as well as tools for defining optimization variables
and objectives.

The optimization module is divided into three subcategories:

1. Optimization Core Functionalities
2. Optimization Operands
3. Optimization Variables

Core Functionalities
--------------------

.. autosummary::
   :toctree: optimization/
   :caption: Optimization Modules

   optimization.optimization


Operands
--------

The `optimization.operand` subpackage contains the following modules:

.. autosummary::
   :toctree: optimization/operand/
   :caption: Operand Modules
   :recursive:

   optimization.operand.aberration
   optimization.operand.operand
   optimization.operand.paraxial
   optimization.operand.ray


Variables
---------

The `optimization.variable` subpackage contains the following modules:

.. autosummary::
   :toctree: optimization/variable/
   :caption: Variable Submodules
   :recursive:

   optimization.variable.asphere_coeff
   optimization.variable.base
   optimization.variable.chebyshev_coeff
   optimization.variable.conic
   optimization.variable.decenter
   optimization.variable.index
   optimization.variable.polynomial_coeff
   optimization.variable.radius
   optimization.variable.reciprocal_radius
   optimization.variable.thickness
   optimization.variable.tilt
   optimization.variable.variable
