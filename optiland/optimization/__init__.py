# flake8: noqa

from .variable import (
    VariableBehavior,
    RadiusVariable,
    ConicVariable,
    ThicknessVariable,
    IndexVariable,
    AsphereCoeffVariable,
    PolynomialCoeffVariable,
    ChebyshevCoeffVariable,
    ScipyVariable,
)
from .operand import ParaxialOperand, AberrationOperand, RayOperand, Operand
from .problem import OptimizationProblem
from .optimizer.scipy import (
    OptimizerGeneric,
    LeastSquares,
    DualAnnealing,
    DifferentialEvolution,
    SHGO,
    BasinHopping,
)

try:
    from .optimizer.torch import TorchAdamOptimizer
except (ImportError, ModuleNotFoundError):
    pass
from .optimizer.glass_expert import GlassExpert
