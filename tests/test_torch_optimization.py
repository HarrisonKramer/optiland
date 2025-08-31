import warnings

import pytest

import optiland.backend as be
from optiland.optimization import TorchAdamOptimizer, TorchSGDOptimizer
from optiland.samples.objectives import CookeTriplet
from optiland.optimization import OptimizationProblem


def setup_problem(
    add_variable=True,
    add_operand=True,
    min_val=1.0,
    max_val=10.0,
    target=12.0,
    initial_value=5.0,
):
    """
    Helper function to set up a standard optimization problem for testing.

    Args:
        add_variable (bool): Whether to add a variable to the problem.
        add_operand (bool): Whether to add an operand to the problem.
        min_val (float, optional): The minimum bound for the variable.
        max_val (float, optional): The maximum bound for the variable.
        target (float): The target value for the operand.
        initial_value (float): The initial value for the variable.

    Returns:
        tuple: A tuple containing the OptimizationProblem and the lens object.
    """
    lens = CookeTriplet()
    problem = OptimizationProblem()

    # Set the initial state before adding variables to ensure it's captured
    if add_variable:
        lens.surface_group.surfaces[1].radius = initial_value
        problem.add_variable(
            lens,
            "radius",
            surface_number=1,
            min_val=min_val,
            max_val=max_val,
        )

    if add_operand:
        problem.add_operand(
            operand_type="f2",
            target=target,
            weight=1.0,
            input_data={"optic": lens},
        )

    # Manually update optics to reflect initial state
    problem.update_optics()
    return problem, lens


@pytest.fixture(scope="module", autouse=True)
def set_torch_backend():
    """
    Fixture to ensure the torch backend is set for all tests in this file.
    It will set the backend to torch before the tests run and revert to the
    original backend after all tests in the module are completed.
    """
    original_backend = be.get_backend()
    be.set_backend("torch")
    yield
    be.set_backend(original_backend)


class TestTorchBaseOptimizerSetup:
    """
    Tests focused on the setup and edge cases of the TorchBaseOptimizer,
    which are independent of the specific optimizer algorithm used.
    """

    def test_init_raises_error_if_backend_not_torch(self):
        """
        Ensures that initializing a Torch optimizer without the 'torch'
        backend raises a RuntimeError.
        """
        # This test temporarily switches the backend and must restore it
        # to not affect other tests, as the module-scoped fixture will
        # not run between tests.
        original_backend = be.get_backend()
        try:
            be.set_backend("numpy")
            # We need to re-run setup_problem under the numpy backend
            problem, _ = setup_problem()
            with pytest.raises(RuntimeError, match="requires the 'torch' backend"):
                TorchAdamOptimizer(problem)
        finally:
            # Restore the backend for subsequent tests
            be.set_backend(original_backend)

    def test_init_enables_gradient_tracking_with_warning(self):
        """
        Tests that a warning is issued if gradient tracking is disabled
        when the optimizer is initialized, and that it gets enabled.
        """
        # Create the problem first. This may enable grad mode.
        problem, _ = setup_problem()
        
        # Now, explicitly disable grad mode to test the optimizer's __init__.
        if hasattr(be.grad_mode, "disable"):
             be.grad_mode.disable()
        assert not be.grad_mode.requires_grad

        # The optimizer's __init__ should now issue a warning and enable grad mode.
        with pytest.warns(UserWarning, match="Gradient tracking is enabled for PyTorch"):
             optimizer = TorchAdamOptimizer(problem)
        
        # Check that gradients are now enabled by the optimizer
        assert be.grad_mode.requires_grad
        
        # Cleanup: Ensure grad mode is enabled for subsequent tests
        if hasattr(be.grad_mode, "enable"):
            be.grad_mode.enable()

@pytest.mark.parametrize("optimizer_class", [TorchAdamOptimizer, TorchSGDOptimizer])
class TestTorchOptimizers:
    """
    A parametrized test suite for all concrete Torch optimizer implementations.
    """

    def test_optimize_successfully_reduces_loss(self, optimizer_class):
        """
        Verifies that the optimizer successfully reduces the loss function value
        over a number of steps.
        """
        problem, _ = setup_problem()
        initial_loss = problem.sum_squared().item()

        optimizer = optimizer_class(problem)
        result = optimizer.optimize(n_steps=50, disp=False)

        assert result.fun < initial_loss
        assert be.isclose(be.array(result.fun), be.array(problem.sum_squared().item()))

    def test_optimize_no_operands_returns_zero_loss(self, optimizer_class):
        """
        Tests that optimization with no operands results in a loss of zero,
        as there is nothing to optimize.
        """
        problem, _ = setup_problem(add_operand=False)
        optimizer = optimizer_class(problem)
        result = optimizer.optimize(n_steps=10, disp=False)
        assert be.isclose(be.array(result.fun), be.array(0.0))

    def test_optimize_no_variables_raises_error(self, optimizer_class):
        """
        Tests that attempting to optimize a problem with no variables raises
        a ValueError, as there are no parameters for the optimizer to act on.
        """
        problem, _ = setup_problem(add_variable=False)
        optimizer = optimizer_class(problem)
        with pytest.raises(ValueError):
            optimizer.optimize(n_steps=10)

    def test_optimize_with_zero_steps_makes_no_change(self, optimizer_class):
        """
        Tests that running the optimizer for zero steps results in no change
        to the loss function or the variable values.
        """
        problem, _ = setup_problem(initial_value=5.0)
        initial_loss = problem.sum_squared().item()
        initial_param_val = problem.variables[0].variable.get_value()

        optimizer = optimizer_class(problem)
        result = optimizer.optimize(n_steps=0, disp=False)

        assert be.isclose(be.array(result.fun), be.array(initial_loss))
        final_param_val = problem.variables[0].variable.get_value()
        assert be.isclose(initial_param_val, final_param_val)


    def test_callback_is_invoked_at_each_step(self, optimizer_class):
        """
        Ensures the callback function is called exactly n_steps times with the
        correct step index and a float loss value.
        """
        problem, _ = setup_problem()
        optimizer = optimizer_class(problem)
        n_steps = 10
        history = []

        def callback_fn(step, loss):
            history.append((step, loss))

        optimizer.optimize(n_steps=n_steps, callback=callback_fn, disp=False)

        assert len(history) == n_steps
        for i, (step, loss) in enumerate(history):
            assert step == i
            assert isinstance(loss, float)
            if i > 0: # Loss is not guaranteed to decrease every step for all optimizers
                assert loss >= 0.0

    @pytest.mark.parametrize("disp, should_have_output", [(True, True), (False, False)])
    def test_display_output_controlled_by_disp_flag(self, capsys, optimizer_class, disp, should_have_output):
        """
        Tests that console output is correctly controlled by the 'disp' flag.
        """
        problem, _ = setup_problem()
        optimizer = optimizer_class(problem)
        optimizer.optimize(n_steps=1, disp=disp)
        captured = capsys.readouterr()

        if should_have_output:
            assert "Loss" in captured.out
        else:
            assert "Loss" not in captured.out
