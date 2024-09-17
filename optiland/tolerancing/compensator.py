from optiland.optimization import (
    OptimizationProblem,
    OptimizerGeneric,
    LeastSquares,
    DualAnnealing,
    DifferentialEvolution)


class CompensatorOptimizer(OptimizationProblem):
    """
    A class representing a compensator for a tolerancing problem. This class
    optimizes a set of variables to compensate for perturbations in an optical
    system.

    Args:
        optimizer (str): The type of optimizer to use. Default is 'generic'.
            Other options include 'least_squares', 'dual_annealing', and
            'differential_evolution'.
        **kwargs: Additional keyword arguments to be passed to the optimizer.

    Attributes:
        optimizer (str): The type of optimizer being used.
        kwargs (dict): Additional keyword arguments passed to the optimizer.
        _optimizer_map (dict): A mapping of optimizer types to their
            respective classes.
    """

    def __init__(self, optimizer='generic', **kwargs):
        super().__init__()
        self.optimizer = optimizer
        self.kwargs = kwargs

        self._optimizer_map = {
            'generic': OptimizerGeneric,
            'least_squares': LeastSquares,
            'dual_annealing': DualAnnealing,
            'differential_evolution': DifferentialEvolution
        }

    @property
    def has_variables(self):
        """
        Check if the optimizer has variables. If no variables are present,
        then the optimizer will not run.

        Returns:
            bool: True if the optimizer has variables, False otherwise.
        """
        return bool(self.variables)

    def run(self):
        """
        Run the optimizer for compensation.

        Returns:
            scipy.optimize.OptimizeResult: The result of the optimizer run.
        """
        optimizer = self._optimizer_map[self.optimizer](**self.kwargs)
        return optimizer.run(self)
