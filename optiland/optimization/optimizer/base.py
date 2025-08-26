"""Optimizer Base Module

This module contains the abstract base class for an optimizer in an optimization
process. The AbstractOptimizer class is an abstract class that represents the
behavior of an optimizer. It is used as a base class for all optimizers in the
optimization process.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Abstract base class for an optimization algorithm.

    This class defines the common interface for all optimizers, ensuring they
    can be used interchangeably.
    """

    def __init__(self, problem):
        """
        Initializes the optimizer with an OptimizationProblem.

        Args:
            problem (OptimizationProblem): The optimization problem to be solved.
        """
        self.problem = problem

    @abstractmethod
    def optimize(self, **kwargs):
        """
        Run the optimization algorithm.

        This method must be implemented by all concrete optimizer subclasses.
        """
        pass  # pragma: no cover
