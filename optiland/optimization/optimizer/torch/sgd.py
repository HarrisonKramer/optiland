"""Torch SGD Optimizer

This module contains an implementation of the SGD optimizer for PyTorch.

Kramer Harrison, 2025
"""

from __future__ import annotations

try:
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import ExponentialLR, LRScheduler
except (ImportError, ModuleNotFoundError):
    torch = None
    optim = None
    ExponentialLR = None
    LRScheduler = None

from .base import TorchBaseOptimizer


class TorchSGDOptimizer(TorchBaseOptimizer):
    """
    An optimizer that uses the PyTorch SGD algorithm.

    This optimizer leverages automatic differentiation to perform gradient-based
    optimization on an OptimizationProblem.
    """

    def _create_optimizer_and_scheduler(
        self, lr: float, gamma: float
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """
        Creates and returns the SGD optimizer and an ExponentialLR scheduler.

        Args:
            lr (float): The learning rate.
            gamma (float): The decay factor for the learning rate.

        Returns:
            tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]: The
                optimizer and learning rate scheduler.
        """
        optimizer = optim.SGD(self.params, lr=lr)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        return optimizer, scheduler
