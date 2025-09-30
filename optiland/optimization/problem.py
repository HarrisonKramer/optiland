"""Optimization Problem Module

This module contains the OptimizationProblem class, which represents an
optimization problem. The class allows for the addition of operands and
variables to the merit function, and provides methods to evaluate the merit
function and print information about the optimization problem.

Kramer Harrison, 2025
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd

import optiland.backend as be
from optiland.optimization.operand import OperandManager
from optiland.optimization.variable import VariableManager

if TYPE_CHECKING:
    from optiland.optimization.scaling.base import Scaler


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

        # Enable gradient tracking for PyTorch
        if be.get_backend() == "torch" and not be.grad_mode.requires_grad:
            warnings.warn("Gradient tracking is enabled for PyTorch.", stacklevel=2)
            be.grad_mode.enable()

    @staticmethod
    def _to_item(x):
        """
        Convert a single-element backend array to a Python scalar.
        This is a utility for printing and string formatting.
        """
        if x is None:
            return None
        if hasattr(x, "item"):
            return x.item()
        return x

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

    def add_variable(self, optic, variable_type, scaler: Scaler = None, **kwargs):
        """Add a variable to the merit function"""
        self.variables.add(optic, variable_type, scaler=scaler, **kwargs)

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
        terms = [op.fun() for op in self.operands]
        if not terms:
            return be.array([0.0])
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
                f"{self._to_item(op.target):+.3f}" if op.target is not None else ""
                for op in self.operands
            ],
            "Min. Bound": [
                self._to_item(op.min_val) if op.min_val is not None else ""
                for op in self.operands
            ],
            "Max. Bound": [
                self._to_item(op.max_val) if op.max_val is not None else ""
                for op in self.operands
            ],
            "Weight": [self._to_item(op.weight) for op in self.operands],
            "Value": [f"{self._to_item(op.value):+.3f}" for op in self.operands],
            "Delta": [f"{self._to_item(op.delta()):+.3f}" for op in self.operands],
        }

        df = pd.DataFrame(data)
        values = self.fun_array()
        total = be.sum(values)

        total_item = self._to_item(total)

        if total_item == 0.0:
            df["Contrib. [%]"] = 0.0
        else:
            contrib = be.round(values / total * 100, decimals=2)
            df["Contrib. [%]"] = be.to_numpy(contrib)

        print(df.to_markdown(headers="keys", tablefmt="fancy_outline"))

    def variable_info(self):
        """Print information about the variables in the merit function."""
        data = {
            "Variable Type": [var.type for var in self.variables],
            "Surface": [var.surface_number for var in self.variables],
            "Value": [
                self._to_item(var.variable.inverse_scale(var.value))
                for var in self.variables
            ],
            "Min. Bound": [self._to_item(var.min_val) for var in self.variables],
            "Max. Bound": [self._to_item(var.max_val) for var in self.variables],
        }

        df = pd.DataFrame(data)
        print(df.to_markdown(headers="keys", tablefmt="fancy_outline"))

    def merit_info(self):
        """Print information about the merit function."""
        current_value = self.sum_squared()

        # Convert tensor to a Python scalar for calculations and printing
        printable_current_value = self._to_item(current_value)

        if self.initial_value == 0.0:
            improve_percent = 0.0
        else:
            improve_percent = (
                (self.initial_value - printable_current_value)
                / self.initial_value
                * 100
            )

        data = {
            "Merit Function Value": [printable_current_value],
            "Improvement (%)": improve_percent,
        }
        df = pd.DataFrame(data)
        print(df.to_markdown(headers="keys", tablefmt="fancy_outline"))

    def info(self):
        """Print information about the optimization problem."""
        self.merit_info()
        self.operand_info()
        self.variable_info()
