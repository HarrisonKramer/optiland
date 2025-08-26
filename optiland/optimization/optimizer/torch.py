"""Torch Optimizer

This module contains the TorchAdamOptimizer class, which implements the Adam
optimization algorithm using the PyTorch backend.

Kramer Harrison, 2025
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

import optiland.backend as be

from ..variable.torch import TorchVariable
from .base import BaseOptimizer

if TYPE_CHECKING:
    from ..problem import OptimizationProblem


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

        # Create a parallel list of TorchVariables from the problem definition
        self.torch_variables = []
        for var_def in self.problem.variables:
            tv = TorchVariable(
                optic=var_def.optic,
                surface_number=getattr(var_def, "surface_number", None),
                initial_value=var_def.value,
                apply_scaling=var_def.apply_scaling,
            )
            self.torch_variables.append(tv)

    def _sync_params_to_optics(self):
        """
        Pushes the current tensor values from TorchVariables into the optic objects.
        """
        for i, torch_var in enumerate(self.torch_variables):
            original_var = self.problem.variables[i]
            original_var.variable.update_value(torch_var.value)

    def _apply_bounds(self):
        """
        Applies the defined bounds to the parameters in-place.
        This is called after each optimizer step to enforce constraints.
        """
        with torch.no_grad():
            for i, torch_var in enumerate(self.torch_variables):
                min_val, max_val = self.problem.variables[i].bounds
                if min_val is not None and max_val is not None:
                    torch_var.value.data.clamp_(min_val, max_val)
                elif min_val is not None:
                    torch_var.value.data.clamp_(min=min_val)
                elif max_val is not None:
                    torch_var.value.data.clamp_(max=max_val)

    def optimize(self, n_steps: int = 100, lr: float = 1e-2, disp: bool = True):
        """
        Runs the Adam optimization loop.

        Args:
            n_steps (int): The number of optimization steps to perform.
            lr (float): The learning rate for the Adam optimizer.
            disp (bool): If True, prints the loss at regular intervals.
        """
        params = [tv.value for tv in self.torch_variables]
        optimizer = torch.optim.Adam(params, lr=lr)

        with be.grad_mode.temporary_enable():
            for i in range(n_steps):
                optimizer.zero_grad()

                # 1. Sync Tensors -> Optic State
                self._sync_params_to_optics()

                # 2. Compute Loss
                # Triggers the forward pass through the differentiable model
                loss = self.problem.sum_squared()

                # 3. Backpropagate and Optimize
                loss.backward()
                optimizer.step()

                # 4. Enforce Constraints
                self._apply_bounds()

                if disp and (i % 10 == 0 or i == n_steps - 1):
                    print(f"  Step {i:04d}/{n_steps - 1}, Loss: {loss.item():.6f}")

        # Final sync to ensure the optic objects have the optimized values
        self._sync_params_to_optics()
        self.problem.update_optics()

        # Return a result object similar to scipy's OptimizeResult for compatibility
        final_loss = self.problem.sum_squared().item()
        return SimpleNamespace(fun=final_loss, x=[p.item() for p in params])
