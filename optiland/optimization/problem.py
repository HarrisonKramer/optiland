"""Optimization Problem Module

This module contains the OptimizationProblem class, which represents an
optimization problem. The class allows for the addition of operands and
variables to the merit function, and provides methods to evaluate the merit
function and print information about the optimization problem.

Kramer Harrison, 2025
"""

from __future__ import annotations

import pandas as pd

import optiland.backend as be
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
        terms = [op.fun() for op in self.operands]  # each should be a torch scalar
        return be.stack(terms) ** 2

    def sum_squared(self):
        """Calculate the sum of squared operand weighted deltas"""
        return be.sum(self.fun_array())

    def rss(self):
        """RSS of current merit function"""
        return be.sqrt(self.sum_squared())

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
        total = be.sum(values)
        if total == 0.0:
            df["Contrib. [%]"] = 0.0
        else:
            df["Contrib. [%]"] = be.round(values / total * 100, decimals=2)

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
