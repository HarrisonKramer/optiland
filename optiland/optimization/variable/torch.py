"""Torch Variable

This module contains the TorchVariable class, which is a wrapper around
torch.nn.Parameter for use in the optimization framework.

Kramer Harrison, 2025
"""

from __future__ import annotations

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

import optiland.backend as be

from .base import VariableBehavior


class TorchVariable(VariableBehavior):
    """
    Represents a variable that wraps a torch.nn.Parameter.

    This class holds the differentiable state for a variable when using the
    PyTorch backend. It is instantiated and managed by a Torch-compatible
    optimizer.
    """

    def __init__(self, optic, surface_number, initial_value, apply_scaling=True):
        if be.get_backend() != "torch":
            raise ValueError("TorchVariable can only be used with the PyTorch backend.")

        super().__init__(optic, surface_number, apply_scaling)
        self._param = torch.nn.Parameter(be.array(initial_value))

    @property
    def value(self):
        """Returns the underlying torch.nn.Parameter."""
        return self._param

    def get_value(self):
        """Compatibility method to get the tensor value."""
        return self.value

    def update_value(self, new_value):
        """
        Updates the parameter's data in-place without tracking gradients.

        This is useful for resetting the variable to a specific state.
        """
        if isinstance(new_value, torch.nn.Parameter):
            # bind to the same object so optimizer and variable share it
            self._param = new_value
        else:
            # previous behavior: copy data into existing parameter
            with torch.no_grad():
                new_tensor = be.array(new_value)
                self.value.data.copy_(new_tensor)
