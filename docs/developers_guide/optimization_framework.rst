Optimization Framework
======================

Optimization in Optiland allows users to improve the performance of optical systems by adjusting design parameters to minimize
(or maximize) a merit function. The framework supports a wide range of optimizers and a flexible system for defining operands and
variables. The framework integrates tightly with the `Optic` class.

The Optiland optimization framework includes the following components:

- **Operands**: Quantitative metrics for evaluating optical system performance or properties (e.g., RMS spot size, wavefront error, etc.).
- **Variables**: System parameters that can be adjusted (e.g., surface curvatures, separations).
- **Optimization Problem Class**: Encapsulates the problem definition, linking operands and variables.
- **Optimizers**: Algorithms for solving optimization problems, such as gradient-based methods or evolutionary strategies.


Components Explained
--------------------

1. **Optimization Problem**:

   - The `OptimizationProblem` class orchestrates the optimization process.
   - Key responsibilities include:

     - Adding **operands** to define the merit function.
     - Adding **variables** to define the parameters to optimize.
     - Computing the overall objective function value.

2. **Optimizers**:

   - A base `Optimizer` class wraps `scipy.optimize.minimize` and provides a unified interface.
   - Built-in optimizers include:

     - **Dual Annealing** (global)
     - **Differential Evolution** (global)
     - **Basin Hopping** (global)
     - **SHGO** (global)
     - **Least Squares** (local)
     - **Nelder-Mead**, **Powell**, **BFGS**, **L-BFGS-B**, **COBYLA**, etc. (local optimization, from `scipy.optimize.minimize`)
   - Users can subclass the base optimizer for custom methods.

3. **Operands and Variables**:

   - **Operands**: Define individual contributions to the merit function. Examples:

     - RMS Spot Size
     - Wavefront Error
     - Focal Length
   - **Variables**: Define the parameters to optimize, such as:

     - Radius of curvature
     - Conic constants
     - Material refractive indices
     - Surface tilts and decenters
     - Geometry parameters (e.g., freeform coefficients)

.. note::
   The optimization framework is written in a modular way, allowing users to easily extend the framework with custom optimizers, operands, and variables.


Typical Optimization Process
----------------------------

1. **Set Up the Problem**: Create an instance of `OptimizationProblem`:

.. code:: python

   from optiland.optimization import OptimizationProblem
   problem = OptimizationProblem(optic)

2. **Add Operands**: Add operands to define the merit function:

.. code:: python

   input_data = {'optic': lens}

   # Add focal length operand
   problem.add_operand(operand_type='f2', target=50, weight=1, input_data=input_data)

3. **Add Variables**: Define the parameters to optimize:

.. code:: python

   # Add radius of curvature variable for second surface
   problem.add_variable(lens, 'radius', surface_number=2)

4. **Choose an Optimizer**: Select an optimizer and run the optimization:

.. code:: python

   from optiland.optimization import OptimizerGeneric
   optimizer = OptimizerGeneric(problem)
   result = optimizer.optimize()

5. **Review Results**: Print optimization results and visualize performance:

.. code:: python

   problem.info()  # print optimization problem details
   print(result)  # standard output from scipy.optimize.minimize

Understanding Operands
----------------------

Operands represent individual components of the merit function. To find the inputs required for a specific operand:

- Refer to the operand registry in the Operand module, or the API documentation.
- Use operand-specific documentation for parameter details. For example, the RMS spot size requires a field as an input, while the focal length does not. All operands require a target value, weight, and an `Optic` instance.

Extending Optimization
----------------------

Custom operands, variables and optimization algorithms can be added by subclassing the appropriate base classes. For example:

- Subclass VariableBehavior to create a new variable type, then register it within the Variable class.
- Define a new operand function and register it within the Operand module.
- Subclass OptimizerGeneric to create a new optimization algorithm.

.. tip::
   See the :ref:`Learning Guide <example_gallery>` for demonstrations of custom optimization algorithms and user-defined operands.
