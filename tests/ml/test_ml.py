# tests/ml/test_ml.py
"""
Tests for the machine learning wrappers in optiland.ml.wrappers.
"""
from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None

import optiland.backend as be
from optiland.ml.wrappers import OpticalSystemModule
from optiland.optimization import OptimizationProblem
from optiland.samples.objectives import CookeTriplet


def setup_problem(
    add_variable=True, add_operand=True, min_val=1.0, max_val=10.0, target=12.0
):
    """
    Helper function to set up a standard optimization problem for testing.

    Args:
        add_variable: Whether to add a variable to the problem.
        add_operand: Whether to add an operand to the problem.
        min_val: The minimum bound for the variable.
        max_val: The maximum bound for the variable.
        target: The target value for the operand.

    Returns:
        A tuple containing the OptimizationProblem and the lens object.
    """
    lens = CookeTriplet()
    problem = OptimizationProblem(optic=lens)
    if add_variable:
        problem.add_variable("thickness", 1, min_val=min_val, max_val=max_val)
    if add_operand:
        problem.add_operand("f2", target=target)
    return problem, lens


@pytest.fixture(scope="module", autouse=True)
def set_torch_backend():
    """
    A module-scoped fixture that ensures the torch backend is active for all
    tests in this file. It reverts to the original backend after all tests
    are completed.
    """
    original_backend = be.get_backend()
    be.set_backend("torch")
    yield
    be.set_backend(original_backend)


class TestOpticalSystemModule:
    """
    Tests for the OpticalSystemModule, which wraps an optiland OptimizationProblem
    into a PyTorch nn.Module for integration with ML workflows.
    """

    def test_init_runtime_error_wrong_backend(self):
        """
        Tests that a RuntimeError is raised if the backend is not 'torch' when
        initializing the module.
        """
        original_backend = be.get_backend()
        try:
            be.set_backend("numpy")
            problem, lens = setup_problem()
            with pytest.raises(RuntimeError, match="requires the 'torch' backend"):
                OpticalSystemModule(problem)
        finally:
            be.set_backend(original_backend)

    def test_init_enables_gradients(self):
        """
        Tests that a warning is issued and gradients are enabled if they are
        initially disabled.
        """
        problem, _ = setup_problem()
        be.grad_mode.disable()
        with pytest.warns(UserWarning, match="Gradient tracking is enabled"):
            _ = OpticalSystemModule(problem)
        assert be.grad_mode.requires_grad
        be.grad_mode.enable()  # Restore for subsequent tests

    def test_init_parameter_creation(self):
        """
        Tests that the module's `nn.ParameterList` is created correctly and
        initialized with values from the problem's variables.
        """
        problem, _ = setup_problem()
        module = OpticalSystemModule(problem)
        assert isinstance(module.params, nn.ParameterList)
        assert len(module.params) == len(problem.variables)
        initial_val = problem.variables[0].value
        assert be.isclose(module.params[0].data, be.array(initial_val))

    def test_default_loss_function(self):
        """
        Tests that the default loss function correctly computes the sum of
        squared errors from the optimization problem.
        """
        problem, _ = setup_problem()
        module = OpticalSystemModule(problem)
        expected_loss = problem.sum_squared()
        actual_loss = module._default_loss()
        assert be.isclose(expected_loss, actual_loss)

    def test_sync_params_to_problem_and_bounds(self):
        """
        Tests that the module's parameters are correctly synced to the
        underlying problem, and that variable bounds are enforced.
        """
        min_b, max_b = 5.0, 15.0
        problem, _ = setup_problem(min_val=min_b, max_val=max_b)
        module = OpticalSystemModule(problem)

        # Test clamping to the maximum bound
        with torch.no_grad():
            module.params[0].data.fill_(20.0)
        module._sync_params_to_problem()
        module.apply_bounds()
        assert be.isclose(module.params[0].data, be.array(max_b))

        # Test clamping to the minimum bound
        with torch.no_grad():
            module.params[0].data.fill_(1.0)
        module._sync_params_to_problem()
        module.apply_bounds()
        assert be.isclose(module.params[0].data, be.array(min_b))

    def test_forward_pass_and_optimization(self):
        """
        Tests a full forward and backward pass, ensuring that the loss
        decreases after an optimization step.
        """
        problem, _ = setup_problem()
        module = OpticalSystemModule(problem)
        initial_loss = be.copy(module.forward())
        optimizer = optim.Adam(module.parameters(), lr=10)

        for _ in range(10):
            optimizer.zero_grad()
            loss = module.forward()
            loss.backward()
            optimizer.step()
            module.apply_bounds()

        final_loss = module.forward()
        assert final_loss.item() < initial_loss.item()

    def test_custom_objective_function(self):
        """
        Tests that a custom objective function can be passed to and used by
        the module for the forward pass.
        """
        problem, _ = setup_problem()
        custom_fn = lambda: torch.tensor(123.45, dtype=torch.float64)
        module = OpticalSystemModule(problem, objective_fn=custom_fn)
        loss = module.forward()
        assert be.isclose(loss, torch.tensor(123.45, dtype=torch.float64))