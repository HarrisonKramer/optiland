import numpy as np
from scipy.optimize import minimize
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
        for op in self.operands:
            op.info()
        print('  Variables: ')
        for var in self.variables:
            var.info()


class OptimizerGeneric:

    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
        self._x = []

    def optimize(self, maxiter=1000, disp=True):

        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])

        def fun(x):
            for idvar, var in enumerate(self.problem.variables):
                var.update(x[idvar])
            funs = np.array([op.fun() for op in self.problem.operands])
            return np.sum(funs**2)

        options = {'maxiter': maxiter, 'disp': disp}

        result = minimize(fun, x0, bounds=bounds, options=options)
        return result

    def undo(self):
        if len(self._x) > 0:
            x0 = self._x[-1]
            for idvar, var in enumerate(self.problem.variables):
                var.update(x0[idvar])
            self._x.pop(-1)
