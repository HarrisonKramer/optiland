"""Torch Base Optimizer

This module contains a base class for all PyTorch-based optimizers.

Kramer Harrison, 2025
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import TYPE_CHECKING

try:
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import ExponentialLR, LRScheduler
except (ImportError, ModuleNotFoundError):
    torch = None
    optim = None
    ExponentialLR = None
    LRScheduler = None

import optiland.backend as be

from ..base import BaseOptimizer

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...problem import OptimizationProblem


class TorchBaseOptimizer(BaseOptimizer, ABC):
    """
    A base class for all PyTorch-based optimizers.

    This class handles the common setup and optimization loop logic for any
    optimizer using the PyTorch backend.
    """

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)
        if be.get_backend() != "torch":
            raise RuntimeError(
                f"{self.__class__.__name__} requires the 'torch' backend."
            )

        # Ensure gradients are enabled for PyTorch operations
        if not be.grad_mode.requires_grad:
            warnings.warn("Gradient tracking is enabled for PyTorch.", stacklevel=2)
            be.grad_mode.enable()

        # Initialize parameters as torch.nn.Parameter objects
        initial_params = [var.variable.get_value() for var in self.problem.variables]
        self.params = [torch.nn.Parameter(be.array(p)) for p in initial_params]

    @abstractmethod
    def _create_optimizer_and_scheduler(
        self, lr: float, gamma: float
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """
        Creates and returns the specific PyTorch optimizer and learning rate scheduler.
        Subclasses must implement this method.

        Args:
            lr (float): The learning rate.
            gamma (float): The decay factor for the learning rate.

        Returns:
            tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]: The
                optimizer and learning rate scheduler.
        """
        raise NotImplementedError

    def _apply_bounds(self):
        """
        Applies the defined bounds to the parameters in-place.
        This is called after each optimizer step to enforce constraints.
        """
        with torch.no_grad():
            for i, param in enumerate(self.params):
                var = self.problem.variables[i]
                min_val, max_val = var.bounds

                # Clamp the parameter data to the defined bounds
                if min_val is not None and max_val is not None:
                    param.data.clamp_(min_val, max_val)
                elif min_val is not None:
                    param.data.clamp_(min=min_val)
                elif max_val is not None:
                    param.data.clamp_(max=max_val)

    def optimize(
        self,
        n_steps: int = 100,
        lr: float = 1e-2,
        gamma: float = 0.99,
        disp: bool = True,
        callback: Callable[[int, float], None] | None = None,
    ):
        """
        Runs the optimization loop.

        Args:
            n_steps (int): The number of optimization steps.
            lr (float): The learning rate.
            gamma (float): The decay factor for the learning rate.
            disp (bool): Whether to display progress.
            callback (Callable[[int, float], None] | None): A callback function to
                be called after each step with the current step and loss value.
        """
        optimizer, scheduler = self._create_optimizer_and_scheduler(lr, gamma)

        with be.grad_mode.temporary_enable():
            for i in range(n_steps):
                optimizer.zero_grad()

                # The logic is unified. The evaluator handles variable
                # updates internally. The `self.params` are passed to the
                # evaluator, which then updates the problem's variables
                # before computing the loss

                if self.problem.use_batched_evaluator:
                    # Use the batched evaluator
                    loss = self.problem.evaluator.evaluate(self.params)
                else:
                    # Fallback to the original, non-batched method
                    for k, param in enumerate(self.params):
                        self.problem.variables[k].variable.update_value(param)
                    self.problem.update_optics()
                    loss = self.problem.sum_squared()

                # Backpropagate and step
                loss.backward()
                optimizer.step()

                # Enforce constraints on the scaled parameters
                self._apply_bounds()

                # Step the learning rate scheduler
                scheduler.step()

                # Call the user-provided callback
                if callback:
                    callback(i, loss.item())

                # Print loss if display is enabled
                if disp and (i % 10 == 0 or i == n_steps - 1):
                    print(f"  Step {i + 1:04d}/{n_steps}, Loss: {loss.item():.6f}")

        # Final update to ensure the model reflects the last optimized state
        for k, param in enumerate(self.params):
            self.problem.variables[k].variable.update_value(param)
        self.problem.update_optics()

        final_loss = self.problem.sum_squared().item()
        return SimpleNamespace(fun=final_loss, x=[p.item() for p in self.params])
