"""
Optiland Optimization Module

This module provides classes for defining and solving optimization problems
for Optiland Optic instances. It includes an OptimizationProblem class for
defining the problem, and several optimizer classes for solving the problem
using different optimization algorithms.

Kramer Harrison, 2024
"""
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
from optiland.optimization.variable import Variable
from optiland.optimization.operand import Operand


class OptimizationProblem:
    """
    Represents an optimization problem.

    Attributes:
        operands (list): List of operands in the merit function.
        variables (list): List of variables in the merit function.
        initial_value (float): Initial value of the merit function.

    Methods:
        add_operand: Add an operand to the merit function.
        add_variable: Add a variable to the merit function.
        fun_array: Array of operand weighted deltas squared, where the delta
            is the difference between the current and target value.
        sum_squared: Sum of squared operand weighted deltas.
        rss: Root Sum of Squares (RSS) of the current merit function.
        operand_info: Print information about the operands in the merit
            function.
        variable_info: Print information about the variables in the merit
            function.
        info: Print information about the merit function, including operand
            and variable info.
    """

    def __init__(self):
        self.operands = []
        self.variables = []
        self.initial_value = 0.0

    def add_operand(self, operand_type, target, weight=1, input_data={}):
        """Add an operand to the merit function"""
        self.operands.append(Operand(operand_type, target, weight, input_data))

    def add_variable(self, optic, variable_type, **kwargs):
        """Add a variable to the merit function"""
        self.variables.append(Variable(optic, variable_type, **kwargs))

    def clear_operands(self):
        """Clear all operands from the merit function"""
        self.initial_value = 0.0
        self.operands = []

    def clear_variables(self):
        """Clear all variables from the merit function"""
        self.initial_value = 0.0
        self.variables = []

    def fun_array(self):
        """Array of operand weighted deltas squared"""
        return np.array([op.fun() for op in self.operands])**2

    def sum_squared(self):
        """Calculate the sum of squared operand weighted deltas"""
        return np.sum(self.fun_array())

    def rss(self):
        """RSS of current merit function"""
        return np.sqrt(self.sum_squared())

    def update_optics(self):
        """Update all optics considered in the optimization problem"""
        unique_optics = set()
        for var in self.variables:
            unique_optics.add(var.optic)
        for optic in unique_optics:
            optic.update()

    def operand_info(self):
        """Print information about the operands in the merit function"""
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
        """Print information about the variables in the merit function."""
        data = {'Variable Type': [var.type for var in self.variables],
                'Surface': [var.surface_number for var in self.variables],
                'Value': [var.variable.inverse_scale(var.value)
                          for var in self.variables],
                'Min. Bound': [var.min_val for var in self.variables],
                'Max. Bound': [var.max_val for var in self.variables]}

        df = pd.DataFrame(data)
        print(df.to_markdown(headers='keys', tablefmt='psql'))

    def merit_info(self):
        """Print information about the merit function."""
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

    def info(self):
        """Print information about the optimization problem."""
        self.merit_info()
        self.operand_info()
        self.variable_info()


class OptimizerGeneric:
    """
    Generic optimizer class for solving optimization problems.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Attributes:
        problem (OptimizationProblem): The optimization problem to be solved.
        _x (list): List to store the values of the variables during
            optimization.

    Methods:
        optimize(maxiter=1000, disp=True, tol=1e-3): Optimize the problem
            using the specified parameters.
        undo(): Undo the last optimization step.
        _fun(x): Internal function to evaluate the objective function.
    """

    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
        self._x = []

        if self.problem.initial_value == 0.0:
            self.problem.initial_value = self.problem.sum_squared()

    def optimize(self, method=None, maxiter=1000, disp=True, tol=1e-3):
        """
        Optimize the problem using the specified parameters.

        Args:
            method (str, optional): The optimization method to use. Default is
                chosen to be one of BFGS, L-BFGS-B, SLSQP, depending on
                whether contraints or bounds given. Follows
                scipy.optimize.minimize method.
            maxiter (int, optional): Maximum number of iterations.
                Default is 1000.
            disp (bool, optional): Whether to display optimization information.
                Default is True.
            tol (float, optional): Tolerance for convergence. Default is 1e-3.

        Returns:
            result (OptimizeResult): The optimization result.
        """
        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])

        options = {'maxiter': maxiter, 'disp': disp}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.minimize(self._fun,
                                       x0,
                                       method=method,
                                       bounds=bounds,
                                       options=options,
                                       tol=tol)
        return result

    def undo(self):
        """
        Undo the last optimization step.
        """
        if len(self._x) > 0:
            x0 = self._x[-1]
            for idvar, var in enumerate(self.problem.variables):
                var.update(x0[idvar])
            self._x.pop(-1)

    def _fun(self, x):
        """
        Internal function to evaluate the objective function.

        Args:
            x (array-like): The values of the variables.

        Returns:
            rss (float): The residual sum of squares.
        """
        for idvar, var in enumerate(self.problem.variables):
            var.update(x[idvar])
        self.problem.update_optics()  # update all optics (e.g., pickups)
        funs = np.array([op.fun() for op in self.problem.operands])
        rss = np.sum(funs**2)
        if np.isnan(rss):
            return 1e10
        else:
            return rss


class LeastSquares(OptimizerGeneric):
    """
    LeastSquares optimizer class for solving optimization problems using the
    least squares method.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Attributes:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(maxiter=None, disp=False, tol=1e-3): Optimize the problem
            using the least squares method.
    """

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)

    def optimize(self, maxiter=None, disp=False, tol=1e-3):
        """
        Optimize the problem using the least squares method.

        Note:
            The least squares method uses the Trust Region Reflective method.

        Args:
            maxiter (int, optional): Maximum number of iterations.
                Defaults to None.
            disp (bool, optional): Whether to display optimization progress.
                Defaults to False.
            tol (float, optional): Tolerance for termination. Defaults to 1e-3.

        Returns:
            result: The optimization result.
        """

        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)

        lower = [var.bounds[0] if var.bounds[0] is not None
                 else -np.inf for var in self.problem.variables]
        upper = [var.bounds[1] if var.bounds[1] is not None
                 else np.inf for var in self.problem.variables]
        bounds = (lower, upper)

        if disp:
            verbose = 2
        else:
            verbose = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.least_squares(self._fun,
                                            x0,
                                            bounds=bounds,
                                            max_nfev=maxiter,
                                            verbose=verbose,
                                            ftol=tol)
        return result


class DualAnnealing(OptimizerGeneric):
    """
    DualAnnealing is an optimizer that uses the dual annealing algorithm
    to find the minimum of an optimization problem.

    Parameters:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(maxiter=1000, disp=True): Runs the dual annealing algorithm
            to optimize the problem and returns the result.
    """

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)

    def optimize(self, maxiter=1000, disp=True):
        """
        Runs the dual annealing algorithm to optimize the problem.

        Parameters:
            maxiter (int): Maximum number of iterations.
            disp (bool): Whether to display the optimization process.

        Returns:
            result: The result of the optimization.
        """
        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])
        if any(None in bound for bound in bounds):
            raise ValueError('Dual annealing requires all variables'
                             ' have bounds.')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.dual_annealing(self._fun,
                                             bounds=bounds,
                                             maxiter=maxiter,
                                             x0=x0)
        return result


class DifferentialEvolution(OptimizerGeneric):
    """
    Differential Evolution optimizer for solving optimization problems.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(maxiter=1000, disp=True, workers=-1): Runs the differential
            evolution optimization algorithm.
    """

    def __init__(self, problem: OptimizationProblem):
        """
        Initializes a new instance of the DifferentialEvolution class.

        Args:
            problem (OptimizationProblem): The optimization problem to be
                solved.
        """
        super().__init__(problem)

    def optimize(self, maxiter=1000, disp=True, workers=-1):
        """
        Runs the differential evolution optimization algorithm.

        Args:
            maxiter (int): Maximum number of iterations.
            disp (bool): Set to True to display status messages.
            workers (int): Number of parallel workers to use. Set to -1 to use
                all available processors.

        Returns:
            result (OptimizeResult): The optimization result.

        Raises:
            ValueError: If any variable in the problem does not have bounds.
        """
        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])
        if any(None in bound for bound in bounds):
            raise ValueError('Differential evolution requires all variables'
                             ' have bounds.')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            if workers == -1:
                updating = 'deferred'
            else:
                updating = 'immediate'

            result = optimize.differential_evolution(self._fun,
                                                     bounds=bounds,
                                                     maxiter=maxiter,
                                                     x0=x0,
                                                     disp=disp,
                                                     updating=updating,
                                                     workers=workers)
        return result
