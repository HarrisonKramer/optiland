from optiland.optimization import (
    OptimizationProblem,
    OptimizerGeneric,
    LeastSquares,
    DualAnnealing,
    DifferentialEvolution)


class CompensatorOptimizer(OptimizationProblem):
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
        return bool(self.variables)

    def run(self):
        optimizer = self._optimizer_map[self.optimizer](**self.kwargs)
        return optimizer.run(self)
