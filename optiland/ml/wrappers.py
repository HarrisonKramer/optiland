"""Machine Learning Wrappers

This module contains wrappers for integrating Optiland with popular machine learning
frameworks like PyTorch. The OpticalSystemModule class is a key component that allows
users to define and optimize optical systems within the PyTorch ecosystem. The `forward`
method can be customized or overridden to implement specific optical system behaviors.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import torch.nn as nn
except ImportError:
    nn = None

import optiland.backend as be
from optiland.analysis import SpotDiagram
from optiland.optimization.variable.torch import TorchVariable

if TYPE_CHECKING:
    from optiland.optic import Optic
    from optiland.optimization.problem import OptimizationProblem


class OpticalSystemModule(nn.Module):
    """
    A PyTorch nn.Module that wraps an Optiland OptimizationProblem.

    This class exposes the optical system's variables as trainable nn.Parameters,
    allowing the entire system to be integrated as a differentiable layer into
    larger machine learning models.
    """

    def __init__(self, optic: Optic, problem: OptimizationProblem):
        super().__init__()
        if be.get_backend() != "torch":
            raise RuntimeError("OpticalSystemModule requires the 'torch' backend.")

        self.optic = optic
        self.problem = problem

        self._torch_variables = []
        params = []

        for var_def in self.problem.variables:
            tv = TorchVariable(
                optic=var_def.optic,
                surface_number=getattr(var_def, "surface_number", None),
                initial_value=var_def.value,
                apply_scaling=var_def.apply_scaling,
            )
            self._torch_variables.append(tv)
            params.append(tv.value)  # tv.value is an nn.Parameter

        # Register parameters so PyTorch optimizers see them
        self.params = nn.ParameterList(params)

    def _sync_params_to_optics(self):
        """
        Pushes the current tensor values from the nn.Parameters into the optic objects.
        """
        for i, torch_var in enumerate(self._torch_variables):
            original_var = self.problem.variables[i]
            original_var.variable.update_value(torch_var.value)

    def forward(self):
        """
        Defines the forward pass of the optical system.

        This example implementation computes the average RMS spot size over all fields
        and wavelengths, which is a common optimization target. The output is a
        differentiable scalar tensor.

        Note that this implementation is just a starting point. Users are encouraged to
        customize the forward method to suit their specific optimization objectives.
        """
        # 1. Update the physical model from the current nn.Parameter values
        self._sync_params_to_optics()
        self.problem.update_optics()

        # 2. Compute RMS spot size average
        spot = SpotDiagram(self.optic)
        spot_size = be.mean(be.array(spot.rms_spot_radius()))
        return spot_size
