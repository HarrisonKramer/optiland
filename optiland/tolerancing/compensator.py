"""Compensator Module

This module contains a class for compensating perturbations in an optical
system. The CompensatorOptimizer class is used to optimize a set of variables
to compensate for perturbations in an optical system.

Kramer Harrison, 2024
"""

from optiland.optimization import LeastSquares, OptimizationProblem, OptimizerGeneric


class CompensatorOptimizer(OptimizationProblem):
    """A class representing a compensator for a tolerancing problem. This class
    optimizes a set of variables to compensate for perturbations in an optical
    system.

    Args:
        method (str): The type of optimizer to use. Default is 'generic'.
            Other options: 'least_squares'
        tol (float): The tolerance for the optimizer. Default is 1e-5.

    Attributes:
        method (str): The type of optimizer to use. Default is 'generic'.
            Other options include 'least_squares'.
        tol (float): The tolerance for the optimizer. Default is 1e-5.
        _optimizer_map (dict): A mapping of optimizer types to their
            respective classes.

    """

    def __init__(self, method="generic", tol=1e-5):
        super().__init__()
        self.method = method
        self.tol = tol

        self._optimizer_map = {
            "generic": OptimizerGeneric,
            "least_squares": LeastSquares,
        }

    def get_optimizer(self):
        if self.method not in self._optimizer_map:
            raise ValueError(f'Invalid optimizer method "{self.method}".')
        return self._optimizer_map[self.method]

    @property
    def has_variables(self):
        """Check if the optimizer has variables. If no variables are present,
        then the optimizer will not run.

        Returns:
            bool: True if the optimizer has variables, False otherwise.

        """
        return bool(self.variables)

    def run(self):
        """Run the optimizer for compensation.

        Returns:
            scipy.optimize.OptimizeResult: The result of the optimizer run.

        """
        optimizer = self.get_optimizer()(self)
        return optimizer.optimize(tol=self.tol)
