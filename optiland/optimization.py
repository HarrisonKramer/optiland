import numpy as np
import pandas as pd
from scipy import optimize
from optiland.variable import Variable
from optiland.operand import Operand


class OptimizationProblem:

    def __init__(self):
        self.operands = []
        self.variables = []

        self.initial_value = 0.0

    def add_operand(self, operand_type, target, weight=1, input_data={}):
        '''add an operand to the merit function'''
        self.operands.append(Operand(operand_type, target, weight, input_data))

    def add_variable(self, optic, variable_type, **kwargs):
        '''add a variable to the merit function'''
        self.variables.append(Variable(optic, variable_type, **kwargs))

    def fun_array(self):
        '''array of operand target deltas'''
        return np.array([op.fun() for op in self.operands])

    def sum_squared(self):
        return np.sum(np.array(self.fun_array())**2)

    def rss(self):
        '''RSS of current merit function'''
        return np.sqrt(self.sum_squared())

    def operand_info(self):
        data = {'Operand Type': [op.type.replace('_', ' ')
                                 for op in self.operands],
                'Target': [op.target for op in self.operands],
                'Weight': [op.weight for op in self.operands],
                'Value': [op.value for op in self.operands],
                'Delta': [op.delta() for op in self.operands]}

        df = pd.DataFrame(data)
        funs = self.fun_array()
        df['Contribution (%)'] = funs / np.sum(funs) * 100

        print(df.to_markdown(headers='keys', tablefmt='psql'))

    def variable_info(self):
        data = {'Variable Type': [var.type for var in self.variables],
                'Surface': [var.surface_number for var in self.variables],
                'Value': [var.value for var in self.variables],
                'Min. Bound': [var.min_val for var in self.variables],
                'Max. Bound': [var.max_val for var in self.variables]}

        df = pd.DataFrame(data)
        print(df.to_markdown(headers='keys', tablefmt='psql'))

    def info(self):
        '''Print info about merit function'''
        current_value = self.sum_squared()

        if self.initial_value == 0.0:
            improve_percent = 0.0
        else:
            improve_percent = ((self.initial_value - current_value) /
                               self.initial_value * 100)

        data = {'Merit Function Value': [self.sum_squared()],
                'Improvement (%)': improve_percent}
        df = pd.DataFrame(data)
        print(df.to_markdown(headers='keys', tablefmt='psql'))

        self.operand_info()
        self.variable_info()


class OptimizerGeneric:

    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
        self.problem.initial_value = self.problem.sum_squared()
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
            verbose = 0

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
