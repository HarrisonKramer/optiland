"""Optiland Optimization Module

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

from optiland.optimization.operand import OperandManager
from optiland.optimization.variable import VariableManager


class OptimizationProblem:
    """Represents an optimization problem.

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
        self.operands = OperandManager()
        self.variables = VariableManager()
        self.initial_value = 0.0

    def add_operand(
        self,
        operand_type=None,
        target=None,
        min_val=None,
        max_val=None,
        weight=1,
        input_data=None,
    ):
        """Add an operand to the merit function"""
        if input_data is None:
            input_data = {}
        self.operands.add(operand_type, target, min_val, max_val, weight, input_data)

    def add_variable(self, optic, variable_type, **kwargs):
        """Add a variable to the merit function"""
        self.variables.add(optic, variable_type, **kwargs)

    def clear_operands(self):
        """Clear all operands from the merit function"""
        self.initial_value = 0.0
        self.operands.clear()

    def clear_variables(self):
        """Clear all variables from the merit function"""
        self.initial_value = 0.0
        self.variables.clear()

    def fun_array(self):
        """Array of operand weighted deltas squared"""
        return np.array([op.fun() for op in self.operands]) ** 2

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
        data = {
            "Operand Type": [op.operand_type.replace("_", " ") for op in self.operands],
            "Target": [
                f"{op.target:+.3f}" if op.target is not None else ""
                for op in self.operands
            ],
            "Min. Bound": [op.min_val if op.min_val else "" for op in self.operands],
            "Max. Bound": [op.max_val if op.max_val else "" for op in self.operands],
            "Weight": [op.weight for op in self.operands],
            "Value": [f"{op.value:+.3f}" for op in self.operands],
            "Delta": [f"{op.delta():+.3f}" for op in self.operands],
        }

        df = pd.DataFrame(data)
        values = self.fun_array()
        total = np.sum(values)
        if total == 0.0:
            df["Contrib. [%]"] = 0.0
        else:
            df["Contrib. [%]"] = np.round(values / total * 100, 2)

        print(df.to_markdown(headers="keys", tablefmt="fancy_outline"))

    def variable_info(self):
        """Print information about the variables in the merit function."""
        data = {
            "Variable Type": [var.type for var in self.variables],
            "Surface": [var.surface_number for var in self.variables],
            "Value": [var.variable.inverse_scale(var.value) for var in self.variables],
            "Min. Bound": [var.min_val for var in self.variables],
            "Max. Bound": [var.max_val for var in self.variables],
        }

        df = pd.DataFrame(data)
        print(df.to_markdown(headers="keys", tablefmt="fancy_outline"))

    def merit_info(self):
        """Print information about the merit function."""
        current_value = self.sum_squared()

        if self.initial_value == 0.0:
            improve_percent = 0.0
        else:
            improve_percent = (
                (self.initial_value - current_value) / self.initial_value * 100
            )

        data = {
            "Merit Function Value": [current_value],
            "Improvement (%)": improve_percent,
        }
        df = pd.DataFrame(data)
        print(df.to_markdown(headers="keys", tablefmt="fancy_outline"))

    def info(self):
        """Print information about the optimization problem."""
        self.merit_info()
        self.operand_info()
        self.variable_info()


class OptimizerGeneric:
    """Generic optimizer class for solving optimization problems.

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

    def optimize(self, method=None, maxiter=1000, disp=True, tol=1e-3, callback=None):
        """Optimize the problem using the specified parameters.

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
            callback (callable): A callable called after each iteration.

        Returns:
            result (OptimizeResult): The optimization result.

        """
        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])

        options = {"maxiter": maxiter, "disp": disp}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.minimize(
                self._fun,
                x0,
                method=method,
                bounds=bounds,
                options=options,
                tol=tol,
                callback=callback,
            )

        # The last function evaluation is not necessarily the lowest.
        # Update all lens variables to their optimized values
        for idvar, var in enumerate(self.problem.variables):
            var.update(result.x[idvar])
        self.problem.update_optics()

        return result

    def undo(self):
        """Undo the last optimization step."""
        if len(self._x) > 0:
            x0 = self._x[-1]
            for idvar, var in enumerate(self.problem.variables):
                var.update(x0[idvar])
            self._x.pop(-1)

    def _fun(self, x) -> float:
        """Internal function to evaluate the objective function.

        Args:
            x (array-like): The values of the variables.

        Returns:
            rss (float): The residual sum of squares.

        """
        # Update all variables to their new values
        for idvar, var in enumerate(self.problem.variables):
            var.update(x[idvar])

        # Update optics (e.g., pickups and solves)
        self.problem.update_optics()

        # Compute merit function value
        try:
            rss = self.problem.sum_squared()
            if np.isnan(rss):
                return 1e10
            return rss
        except ValueError:
            return 1e10


class LeastSquares(OptimizerGeneric):
    """LeastSquares optimizer class for solving optimization problems using the
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
        """Optimize the problem using the least squares method.

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

        lower = [
            var.bounds[0] if var.bounds[0] is not None else -np.inf
            for var in self.problem.variables
        ]
        upper = [
            var.bounds[1] if var.bounds[1] is not None else np.inf
            for var in self.problem.variables
        ]
        bounds = (lower, upper)

        verbose = 2 if disp else 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.least_squares(
                self._fun,
                x0,
                bounds=bounds,
                max_nfev=maxiter,
                verbose=verbose,
                ftol=tol,
            )
        return result


class DualAnnealing(OptimizerGeneric):
    """DualAnnealing is an optimizer that uses the dual annealing algorithm
    to find the minimum of an optimization problem.

    Parameters
    ----------
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods
    -------
        optimize(maxiter=1000, disp=True): Runs the dual annealing algorithm
            to optimize the problem and returns the result.

    """

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)

    def optimize(self, maxiter=1000, disp=True, callback=None):
        """Runs the dual annealing algorithm to optimize the problem.

        Parameters
        ----------
            maxiter (int): Maximum number of iterations.
            disp (bool): Whether to display the optimization process.
            callback (callable): A callable called after each iteration.

        Returns
        -------
            result: The result of the optimization.

        """
        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])
        if any(None in bound for bound in bounds):
            raise ValueError("Dual annealing requires all variables have bounds.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = optimize.dual_annealing(
                self._fun,
                bounds=bounds,
                maxiter=maxiter,
                x0=x0,
                callback=callback,
            )
        return result


class DifferentialEvolution(OptimizerGeneric):
    """Differential Evolution optimizer for solving optimization problems.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(maxiter=1000, disp=True, workers=-1): Runs the differential
            evolution optimization algorithm.

    """

    def __init__(self, problem: OptimizationProblem):
        """Initializes a new instance of the DifferentialEvolution class.

        Args:
            problem (OptimizationProblem): The optimization problem to be
                solved.

        """
        super().__init__(problem)

    def optimize(self, maxiter=1000, disp=True, workers=-1, callback=None):
        """Runs the differential evolution optimization algorithm.

        Args:
            maxiter (int): Maximum number of iterations.
            disp (bool): Set to True to display status messages.
            workers (int): Number of parallel workers to use. Set to -1 to use
                all available processors.
            callback (callable): A callable called after each iteration.

        Returns:
            result (OptimizeResult): The optimization result.

        Raises:
            ValueError: If any variable in the problem does not have bounds.

        """
        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])
        if any(None in bound for bound in bounds):
            raise ValueError(
                "Differential evolution requires all variables have bounds.",
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            updating = "deferred" if workers == -1 else "immediate"

            result = optimize.differential_evolution(
                self._fun,
                bounds=bounds,
                maxiter=maxiter,
                x0=x0,
                disp=disp,
                updating=updating,
                workers=workers,
                callback=callback,
            )
        return result


class SHGO(OptimizerGeneric):
    """Simplicity Homology Global Optimization (SHGO).

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(workers=-1, *args, **kwargs): Runs the SHGO algorithm.

    """

    def __init__(self, problem: OptimizationProblem):
        """Initializes a new instance of the SHGO class.

        Args:
            problem (OptimizationProblem): The optimization problem to be
                solved.

        """
        super().__init__(problem)

    def optimize(self, workers=-1, callback=None, *args, **kwargs):
        """Runs the SHGO algorithm. Note that the SHGO algorithm accepts the same
        arguments as the scipy.optimize.shgo function.

        Args:
            workers (int): Number of parallel workers to use. Set to -1 to use
                all available CPU processors. Default is -1.
            callback (callable): A callable called after each iteration.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            result (OptimizeResult): The optimization result.

        Raises:
            ValueError: If any variable in the problem does not have bounds.

        """
        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])
        if any(None in bound for bound in bounds):
            raise ValueError("SHGO requires all variables have bounds.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            result = optimize.shgo(
                self._fun,
                bounds=bounds,
                workers=workers,
                callback=callback,
                **kwargs,
            )
        return result


class BasinHopping(OptimizerGeneric):
    """Basin-hopping optimizer for solving optimization problems.

    Args:
        problem (OptimizationProblem): The optimization problem to be solved.

    Methods:
        optimize(maxiter=1000, disp=True, workers=-1): Runs the basin-hopping
            optimization algorithm.

    """

    def __init__(self, problem: OptimizationProblem):
        """Initializes a new instance of the BasinHopping class.

        Args:
            problem (OptimizationProblem): The optimization problem to be
                solved.

        """
        super().__init__(problem)

    def optimize(self, niter=100, callback=None, *args, **kwargs):
        """Runs the basin-hopping algorithm. Note that the basin-hopping
        algorithm accepts the same arguments as the
        scipy.optimize.basinhopping function.

        Args:
            niter (int): Number of iterations to perform. Default is 100.
            callback (callable): A callable called after each iteration.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            result (OptimizeResult): The optimization result.

        Raises:
            ValueError: If any variable in the problem does not have bounds.

        """
        x0 = [var.value for var in self.problem.variables]
        self._x.append(x0)
        bounds = tuple([var.bounds for var in self.problem.variables])
        if not all(x is None for pair in bounds for x in pair):
            raise ValueError("Basin-hopping does not accept bounds.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            result = optimize.basinhopping(
                self._fun,
                x0=x0,
                niter=niter,
                callback=callback,
                **kwargs,
            )
        return result
