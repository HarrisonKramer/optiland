"""Torch Optimizer

This module contains the TorchAdamOptimizer class, which implements the Adam
optimization algorithm using the PyTorch backend.

Kramer Harrison, 2025
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

import optiland.backend as be

from ..base import BaseOptimizer

if TYPE_CHECKING:
    from ...problem import OptimizationProblem


class TorchAdamOptimizer(BaseOptimizer):
    """
    An optimizer that uses the PyTorch Adam algorithm.

    This optimizer leverages automatic differentiation to perform gradient-based
    optimization on an OptimizationProblem.
    """

    def __init__(self, problem: OptimizationProblem):
        super().__init__(problem)
        if be.get_backend() != "torch":
            raise RuntimeError("TorchAdamOptimizer requires the 'torch' backend.")
        else:
            if not be.grad_mode.requires_grad:
                warnings.warn("Gradient tracking is enabled for PyTorch.", stacklevel=2)
                be.grad_mode.enable()

        initial_params = [var.variable.get_value() for var in self.problem.variables]
        self.params = [torch.nn.Parameter(be.array(p)) for p in initial_params]

    def _apply_bounds(self):
        """
        Applies the defined bounds to the parameters in-place.
        This is called after each optimizer step to enforce constraints.
        """
        with torch.no_grad():
            for i, param in enumerate(self.params):
                min_val, max_val = self.problem.variables[i].bounds
                if min_val is not None and max_val is not None:
                    param.data.clamp_(min_val, max_val)
                elif min_val is not None:
                    param.data.clamp_(min=min_val)
                elif max_val is not None:
                    param.data.clamp_(max=max_val)

    def optimize(self, n_steps: int = 100, lr: float = 1e-2, disp: bool = True):
        """
        Runs the Adam optimization loop.

        Args:
            n_steps (int): The number of optimization steps to perform.
            lr (float): The learning rate for the Adam optimizer.
            disp (bool): If True, prints the loss at regular intervals.
        """
        optimizer = torch.optim.Adam(self.params, lr=lr)

        with be.grad_mode.temporary_enable():
            for i in range(n_steps):
                optimizer.zero_grad()

                # 1. Update the model state using the optimizer's current params.
                # This function call rebuilds the computational graph for this step.
                for k, param in enumerate(self.params):
                    self.problem.variables[k].variable.update_value(param)

                # 2. Update any dependent properties.
                self.problem.update_optics()

                # 3. Compute loss from the updated model.
                loss = self.problem.sum_squared()

                # 4. Backpropagate and step.
                loss.backward()
                optimizer.step()

                # 5. Enforce constraints on the scaled parameters.
                self._apply_bounds()

                if disp and (i % 10 == 0 or i == n_steps - 1):
                    print(f"  Step {i:04d}/{n_steps - 1}, Loss: {loss.item():.6f}")

        # Final update to ensure the model reflects the last optimized state
        for k, param in enumerate(self.params):
            self.problem.variables[k].variable.update_value(param)
        self.problem.update_optics()

        final_loss = self.problem.sum_squared().item()
        return SimpleNamespace(fun=final_loss, x=[p.item() for p in self.params])
