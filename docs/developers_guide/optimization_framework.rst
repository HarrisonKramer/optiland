Optimization Framework
======================

The optimization framework is built to refine optical systems based on user-defined objectives. Its components include:

- **Operands**: Quantitative metrics for evaluating optical system performance.
- **Variables**: System parameters that can be adjusted (e.g., surface curvatures, separations).
- **Optimization Problem Class**: Encapsulates the problem definition, linking operands and variables.
- **Optimizers**: Algorithms for solving optimization problems, such as gradient-based methods or evolutionary strategies.

This framework integrates tightly with the `Optic` class to allow for system-level optimizations.
