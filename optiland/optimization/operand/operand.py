"""Operand Module

This module defines various operands to be used during lens optimization. These
include paraxial, real ray, aberrations, wavefront, spot size, and many other
operand types.

In general, to use one of these operands in optimization, you
can simply do the following:
    1. Identify the key in the METRIC_DICT variable, or add your new operand.
    2. Identify the input data dictionary that is required for the calculation
       of this operand.
    3. Add the operand to your optimization.OptimizationProblem instance using
       the add_operand method. Include the operand type, target, weight, and
       the input data (in dict format). See examples.

Kramer Harrison, 2024
"""
from dataclasses import dataclass
from optiland.optimization.operand.paraxial import ParaxialOperand
from optiland.optimization.operand.aberration import AberrationOperand
from optiland.optimization.operand.ray import RayOperand


# Dictionary of operands and their associated functions
METRIC_DICT = {
    'f1': ParaxialOperand.f1,
    'f2': ParaxialOperand.f2,
    'F1': ParaxialOperand.F1,
    'F2': ParaxialOperand.F2,
    'P1': ParaxialOperand.P1,
    'P2': ParaxialOperand.P2,
    'N1': ParaxialOperand.N1,
    'N2': ParaxialOperand.N2,
    'EPD': ParaxialOperand.EPD,
    'EPL': ParaxialOperand.EPL,
    'XPD': ParaxialOperand.XPD,
    'XPL': ParaxialOperand.XPL,
    'magnification': ParaxialOperand.magnification,
    'seidel': AberrationOperand.seidels,
    'TSC': AberrationOperand.TSC,
    'SC': AberrationOperand.SC,
    'CC': AberrationOperand.CC,
    'TCC': AberrationOperand.TCC,
    'TAC': AberrationOperand.TAC,
    'AC': AberrationOperand.AC,
    'TPC': AberrationOperand.TPC,
    'PC': AberrationOperand.PC,
    'DC': AberrationOperand.DC,
    'TAchC': AberrationOperand.TAchC,
    'LchC': AberrationOperand.LchC,
    'TchC': AberrationOperand.TchC,
    'TSC_sum': AberrationOperand.TSC_sum,
    'SC_sum': AberrationOperand.SC_sum,
    'CC_sum': AberrationOperand.CC_sum,
    'TCC_sum': AberrationOperand.TCC_sum,
    'TAC_sum': AberrationOperand.TAC_sum,
    'AC_sum': AberrationOperand.AC_sum,
    'TPC_sum': AberrationOperand.TPC_sum,
    'PC_sum': AberrationOperand.PC_sum,
    'DC_sum': AberrationOperand.DC_sum,
    'TAchC_sum': AberrationOperand.TAchC_sum,
    'LchC_sum': AberrationOperand.LchC_sum,
    'TchC_sum': AberrationOperand.TchC_sum,
    'real_x_intercept': RayOperand.x_intercept,
    'real_y_intercept': RayOperand.y_intercept,
    'real_z_intercept': RayOperand.z_intercept,
    'real_x_intercept_lcs': RayOperand.x_intercept_lcs,
    'real_y_intercept_lcs': RayOperand.y_intercept_lcs,
    'real_z_intercept_lcs': RayOperand.z_intercept_lcs,
    'real_L': RayOperand.L,
    'real_M': RayOperand.M,
    'real_N': RayOperand.N,
    'rms_spot_size': RayOperand.rms_spot_size,
    'OPD_difference': RayOperand.OPD_difference
}


class OperandRegistry:
    """
    A registry to manage operand functions.
    This class allows you to register functions with specific operand names,
    retrieve them, and check if an operand name is registered.

    Attributes:
        _registry (dict): A dictionary to store operand names and their
            associated functions.

    Methods:
        register(name, func):
            Register a function with a specified operand name.
        get(name):
            Retrieve the function associated with an operand name.
        __contains__(name):
            Check if an operand name is registered.
        __repr__():
            Return a string representation of the OperandRegistry.
    """
    def __init__(self):
        self._registry = {}

    def register(self, name, func, overwrite=False):
        """Register a function with a specified operand name.

        Args:
            name (str): The name of the operand.
            func (function): The function to be registered.
            overwrite (bool): Whether to overwrite an existing registration.
        """
        if name in self._registry and not overwrite:
            raise ValueError(f'Operand "{name}" is already registered.')
        self._registry[name] = func

    def get(self, name):
        """Retrieve the function associated with an operand name.

        Args:
            name (str): The name of the operand.
        """
        return self._registry.get(name)

    def __contains__(self, name):
        """Check if an operand name is registered.

        Args:
            name (str): The name of the operand.
        """
        return name in self._registry

    def __repr__(self):
        return f'OperandRegistry({list(self._registry.keys())})'


# Create the global operand registry
operand_registry = OperandRegistry()


# Add all operands to the registry
for name, func in METRIC_DICT.items():
    operand_registry.register(name, func)

@dataclass
class Operand:
    """
    Represents an operand used in optimization calculations.
    If no target is specified, a default is created at the current value.

    Attributes:
        operand_type (str): The type of the operand.
        target (float): The target value of the operand.
        bounds (list): The operand should stay between the bounds (bounded operand).
        more_than (float): The operand should stay above this value (inequality operand).
        less_than (float): The operand should stay below this value (inequality operand).
        weight (float): The weight of the operand.
        input_data (dict): Additional input data for the operand.

    Methods:
        value(): Get the current value of the operand.
        delta_target(): Calculate the difference between the target and current value.
        delta_bounds(): Calculate the difference between the target and the closest bound.
        fun(): Calculate the objective function value.
    """

    operand_type: str = None
    target: float = None
    bounds: list = None
    more_than: float = None,
    less_than: float = None,
    weight: float = None
    input_data: dict = None

    def __post_init__(self):
        if self.bounds is not None and self.bounds[0] >= self.bounds[1]:
            raise ValueError(f"{self.operand_type} operand : lower bound higher than upper bound")
        if self.target is not None and self.bounds is not None:
            raise ValueError(f"{self.operand_type} operand cannot accept both a target and bounds")
        if all(x is None for x in [self.target, self.bounds, self.more_than, self.less_than]):
            # No target has been defined, default it to the current value
            self.target = self.value
        if self.bounds is not None and (self.less_than is not None or self.more_than is not None):
            raise ValueError(f"{self.operand_type} operand cannot accept both bounds and less_than & more_than")
        
    @property
    def value(self):
        """Get current value of the operand"""
        metric_function = operand_registry.get(self.operand_type)
        if metric_function:
            return metric_function(**self.input_data)
        else:
            raise ValueError(f'Unknown operand type: {self.operand_type}')

    def delta_target(self):
        """Calculate the difference between the value and target"""
        return self.value - self.target

    def delta_bounds(self):
        """Calculate the difference between the value and the closest bound"""
        distance_to_closest_bound = min(abs(self.value-self.bounds[0]), abs(self.value-self.bounds[1]))
        return distance_to_closest_bound

    def delta_ineq(self):
        """Calculate the difference between the value and the targets"""
        
        # One of the two
        if self.more_than is not None and not self.less_than:
            return 0 if self.value > self.more_than else self.more_than-self.value
        
        # One of the two
        if self.less_than is not None and not self.more_than:
            return 0 if self.value < self.less_than else self.value-self.less_than
        
        # Both of them
        if self.less_than is not None and self.more_than is not None:
            distance_to_closest_target = min(abs(self.value-self.more_than), abs(self.value-self.less_than))
            return 0 if (self.value > self.more_than and self.value < self.less_than) else distance_to_closest_target
    
    def delta(self):
        """Calculate the difference to target"""
        if self.target is not None:
            return self.delta_target()
        elif self.bounds is not None:
            return self.delta_bounds()
        elif self.more_than is not None or self.less_than is not None:
            return self.delta_ineq()
        else:
            raise ValueError(f"{self.operand_type} operand cannot compute delta")
        
    def fun(self):
        """Calculate the objective function value"""
        return self.weight * self.delta()

    def __str__(self):
        """Return a string representation of the operand"""
        return self.operand_type.replace('_', ' ')
