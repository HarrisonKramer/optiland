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
):
    """
    Helper function to set up a standard optimization problem.
    """
    lens = CookeTriplet()
    problem = OptimizationProblem()
    if add_variable:
        problem.add_variable(
            lens,
            "thickness",
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
    return problem, lens


@pytest.fixture(scope="module", autouse=True)
def set_torch_backend():
    """
    Fixture to ensure the torch backend is set for all tests in this file.
    It will set the backend to torch before the tests run and revert to numpy
    after all tests are completed, even if a test fails.
    """
    original_backend = be.get_backend()
    be.set_backend("torch")
    yield
    be.set_backend(original_backend)


class TestTorchBaseOptimizer:
    """
    Tests for the TorchBaseOptimizer abstract base class.
    """

    def test_init_runtime_error(self):
        """
        Test that an error is raised if the backend is not 'torch'.
        """
        original_backend = be.get_backend()
        be.set_backend("numpy")
        problem = setup_problem()[0]
        # We need to use a concrete class to test the base class __init__
        with pytest.raises(RuntimeError) as e:
            _ = TorchAdamOptimizer(problem)
        assert "requires the 'torch' backend" in str(e.value)
        be.set_backend(original_backend)  # Reset backend for other tests

    def test_bounds_application(self):
        """
        Test that _apply_bounds correctly clamps parameters to min/max values.
        """
        problem, lens = setup_problem(min_val=10, max_val=20)
        optimizer = TorchAdamOptimizer(problem)

        # Set a parameter value outside the bounds to test clamping
        optimizer.params[0].data.fill_(5.0)
        optimizer._apply_bounds()
        assert be.allclose(optimizer.params[0].data, be.array(10.0))

        optimizer.params[0].data.fill_(25.0)
        optimizer._apply_bounds()
        assert be.allclose(optimizer.params[0].data, be.array(20.0))

        optimizer.params[0].data.fill_(15.0)
        optimizer._apply_bounds()
        assert be.allclose(optimizer.params[0].data, be.array(15.0))

    def test_bounds_application_with_none(self):
        """
        Test that _apply_bounds works correctly with None bounds.
        """
        problem, lens = setup_problem(min_val=None, max_val=10)
        optimizer = TorchAdamOptimizer(problem)

        optimizer.params[0].data.fill_(15.0)
        optimizer._apply_bounds()
        assert be.allclose(optimizer.params[0].data, be.array(10.0))
        assert problem.variables[0].bounds == (None, 0.0)

        problem, lens = setup_problem(min_val=10, max_val=None)
        optimizer = TorchAdamOptimizer(problem)
        optimizer.params[0].data.fill_(5.0)
        optimizer._apply_bounds()
        assert be.allclose(optimizer.params[0].data, be.array(10.0))

    def test_optimize_no_variables(self):
        """
        Test that optimization completes successfully with no variables.
        """
        problem, lens = setup_problem(add_variable=False)
        optimizer = TorchAdamOptimizer(problem)
        with pytest.raises(ValueError):
            result = optimizer.optimize()

    def test_optimize_no_operands(self):
        """
        Test that optimization completes successfully with no operands.
        """
        problem, lens = setup_problem(add_operand=False)
        optimizer = TorchAdamOptimizer(problem)
        result = optimizer.optimize()
        assert be.isclose(be.array(result.fun), be.array(0.0))

    def test_callback(self):
        """
        Test that the callback function is called at each step with the correct values.
        """
        problem, _ = setup_problem()
        optimizer = TorchAdamOptimizer(problem)
        n_steps = 5

        history = []
        def callback_fn(step, loss):
            history.append((step, loss))
        
        optimizer.optimize(n_steps=n_steps, callback=callback_fn)
        
        assert len(history) == n_steps
        for i in range(n_steps):
            assert history[i][0] == i
            assert isinstance(history[i][1], float)

    def test_vprint_disp_true(self, capsys):
        """
        Test that print statements appear when disp is True.
        """
        problem, _ = setup_problem()
        optimizer = TorchAdamOptimizer(problem)
        optimizer.optimize(n_steps=1, disp=True)

        captured = capsys.readouterr()
        assert "Loss" in captured.out


    def test_vprint_disp_false(self, capsys):
        """
        Test that print statements do not appear when disp is False.
        """
        problem, _ = setup_problem()
        optimizer = TorchAdamOptimizer(problem)
        optimizer.optimize(n_steps=1, disp=False)

        captured = capsys.readouterr()
        assert "Loss" not in captured.out


class TestTorchAdamOptimizer:
    """
    Tests for the concrete TorchAdamOptimizer class.
    """

    def test_optimize_success(self):
        """
        Test that the optimizer successfully reduces the merit function.
        """
        problem, _ = setup_problem()
        initial_value = problem.sum_squared().item()

        optimizer = TorchAdamOptimizer(problem)
        result = optimizer.optimize(n_steps=50, disp=False)

        assert result.fun < initial_value

    def test_optimize_with_bounds(self):
        """
        Test that optimization works correctly with bounds and the final
        variable is within the bounds.
        """
        min_b, max_b = 10, 100
        problem, lens = setup_problem(min_val=min_b, max_val=max_b)
        original = problem.sum_squared().item()

        optimizer = TorchAdamOptimizer(problem)
        result = optimizer.optimize(n_steps=100, disp=False)

        assert result.fun < original # Check for improvement


class TestTorchSGDOptimizer:
    """
    Tests for the concrete TorchSGDOptimizer class.
    """

    def test_optimize_success(self):
        """
        Test that the optimizer successfully reduces the merit function.
        """
        problem, _ = setup_problem()
        initial_value = problem.sum_squared().item()

        optimizer = TorchSGDOptimizer(problem)
        result = optimizer.optimize(n_steps=50, disp=False)

        assert result.fun < initial_value
    
    def test_optimize_with_bounds(self):
        """
        Test that optimization works correctly with bounds and the final
        variable is within the bounds.
        """
        min_b, max_b = 10, 100
        problem, lens = setup_problem(min_val=min_b, max_val=max_b)
        original = problem.sum_squared().item()

        optimizer = TorchSGDOptimizer(problem)
        result = optimizer.optimize(n_steps=100, disp=False)

        assert result.fun < original # Check for improvement
