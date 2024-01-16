import numpy as np
from scipy import optimize
from optiland.variable import Variable
from optiland.operand import Operand


class OptimizationProblem:

    def __init__(self):
        self.operands = []
        self.variables = []

    def add_operand(self, operand_type, target, weight=1, input_data={}):
        '''add an operand to the merit function'''
        self.operands.append(Operand(operand_type, target, weight, input_data))

    def add_variable(self, optic, variable_type, **kwargs):
        '''add a variable to the merit function'''
        self.variables.append(Variable(optic, variable_type, **kwargs))

    def fun_array(self):
        '''array of operand target deltas'''
        return np.array([op.fun() for op in self.operands])

    def rss(self):
        '''RSS of current merit function'''
        return np.sqrt(np.sum(np.array(self.fun_array())**2))

    def info(self):
        '''Print info about merit function'''
        print('Merit Function Information')
        print(f'  Value: {self.rss()}')
        print('  Operands: ')
        for k, op in enumerate(self.operands):
            op.info(k)
        print('  Variables: ')
        for k, var in enumerate(self.variables):
            var.info(k)


class OptimizerGeneric:

    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
        self._x = []

    def optimize(self, maxiter=1000, disp=True, tol=1e-3):

        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])

        options = {'maxiter': maxiter, 'disp': disp}

        result = optimize.minimize(self._fun,
                                   x0,
                                   bounds=bounds,
                                   options=options,
                                   tol=tol)
        return result

    def undo(self):
        if len(self._x) > 0:
            x0 = self._x[-1]
            for idvar, var in enumerate(self.problem.variables):
                var.update(x0[idvar])
            self._x.pop(-1)

    def _fun(self, x):
        for idvar, var in enumerate(self.problem.variables):
            var.update(x[idvar])
        funs = np.array([op.fun() for op in self.problem.operands])
        return np.sum(funs**2)


class LeastSquares(OptimizerGeneric):

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)

    def optimize(self, maxiter=None, disp=False, tol=1e-3):

        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        lower_bounds = tuple(var.bounds[0] for var in self.problem.variables)
        upper_bounds = tuple(var.bounds[1] for var in self.problem.variables)
        bounds = (lower_bounds, upper_bounds)

        if all(bounds[0] is None and bounds[1] is None for bound in bounds):
            bounds = (-np.inf, np.inf)

        if disp:
            verbose = 2
        else:
            verbose = 1

        result = optimize.least_squares(self._fun,
                                        x0,
                                        bounds=bounds,
                                        max_nfev=maxiter,
                                        verbose=verbose,
                                        ftol=tol)
        return result


class DualAnnealing(OptimizerGeneric):

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)

    def optimize(self, maxiter=1000, disp=True):

        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])
        if any(None in bound for bound in bounds):
            raise ValueError('Dual annealing requires all variables'
                             ' have bounds.')

        result = optimize.dual_annealing(self._fun,
                                         bounds=bounds,
                                         maxiter=maxiter,
                                         x0=x0)
        return result
