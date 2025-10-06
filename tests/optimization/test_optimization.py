# tests/optimization/test_optimization.py
"""
Tests for the optimization framework in optiland.optimization.

This file covers the setup of optimization problems, the functionality of
various optimizer classes, and specialized optimization routines.
"""
import warnings

import optiland.backend as be
import pytest

from optiland.optimization import optimization, glass_expert
from optiland.samples.microscopes import (
    Microscope20x,
    Objective60x,
    UVReflectingMicroscope,
)
from optiland.samples.objectives import CookeTriplet


class TestOptimizationProblem:
    """
    Tests the OptimizationProblem class, which is responsible for setting up
    and managing the components of an optimization task, including operands
    and variables.
    """

    def test_add_operand(self):
        """Tests adding a new operand to the problem."""
        problem = optimization.OptimizationProblem(optic=Objective60x())
        problem.add_operand("f2", target=50, weight=1.0)
        assert len(problem.operands) == 1
        assert problem.operands[0].operand_type == "f2"

    def test_add_variable(self):
        """Tests adding a new variable to the problem."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", surface_number=1, min_val=10, max_val=100)
        assert len(problem.variables) == 1
        assert problem.variables[0].type == "radius"

    def test_clear_operands(self):
        """Tests clearing all operands from the problem."""
        problem = optimization.OptimizationProblem(optic=Objective60x())
        problem.add_operand("f2", target=50)
        problem.clear_operands()
        assert len(problem.operands) == 0

    def test_clear_variables(self):
        """Tests clearing all variables from the problem."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", surface_number=1)
        problem.clear_variables()
        assert len(problem.variables) == 0

    def test_fun_array(self):
        """Tests the calculation of the weighted residual array."""
        problem = optimization.OptimizationProblem(optic=Objective60x())
        problem.add_operand("f2", target=98.57864671748113)
        fun_array = problem.fun_array()
        assert be.allclose(fun_array, be.array([0.0]))

    def test_sum_squared(self):
        """Tests the calculation of the sum of squared weighted residuals."""
        problem = optimization.OptimizationProblem(optic=Objective60x())
        problem.add_operand("f2", target=90)
        sum_squared = problem.sum_squared()
        val = (problem.optic.paraxial.f2() - 90) ** 2
        assert be.isclose(sum_squared, val)

    def test_rss(self):
        """Tests the calculation of the root sum squared merit function."""
        problem = optimization.OptimizationProblem(optic=UVReflectingMicroscope())
        problem.add_operand("f2", target=90)
        rss = problem.rss()
        val = be.abs(problem.optic.paraxial.f2() - 90)
        assert be.isclose(rss, val)

    def test_update_optics(self):
        """
        Tests that the `update_optics` method runs without error. This is
        a smoke test.
        """
        problem = optimization.OptimizationProblem(optic=UVReflectingMicroscope())
        problem.add_operand("f2", target=90)
        problem.update_optics()

    def test_operand_info(self, capsys):
        """Tests that operand information is printed correctly."""
        problem = optimization.OptimizationProblem(optic=UVReflectingMicroscope())
        problem.add_operand("f2", target=9090)
        problem.operand_info()
        captured = capsys.readouterr()
        assert "Operand Type" in captured.out
        assert "Target" in captured.out

    def test_variable_info(self, capsys):
        """Tests that variable information is printed correctly."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", surface_number=1, min_val=10, max_val=100)
        problem.variable_info()
        captured = capsys.readouterr()
        assert "Variable Type" in captured.out
        assert "Value" in captured.out

    def test_merit_info(self, capsys):
        """Tests that merit function information is printed correctly."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", surface_number=1)
        problem.add_operand("f2", target=90)
        problem.merit_info()
        captured = capsys.readouterr()
        assert "Merit Function Value" in captured.out

    def test_info(self, capsys):
        """Tests that the full problem information is printed correctly."""
        problem = optimization.OptimizationProblem(optic=UVReflectingMicroscope())
        problem.add_variable("radius", surface_number=1)
        problem.add_operand("f2", target=90)
        problem.info()
        captured = capsys.readouterr()
        assert "Merit Function Value" in captured.out
        assert "Operand Type" in captured.out
        assert "Variable Type" in captured.out


class TestOptimizerGeneric:
    """
    Tests the base OptimizerGeneric class.
    """

    def test_optimize(self):
        """Tests a simple optimization run."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1, min_val=10, max_val=100)
        problem.add_operand("f2", target=90)
        optimizer = optimization.OptimizerGeneric(problem)
        result = optimizer.optimize(maxiter=10, disp=False, tol=1e-3)
        assert result.success

    def test_undo(self):
        """Tests that the optimizer can undo its changes."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1, min_val=10, max_val=100)
        problem.add_operand("f2", target=90)
        optimizer = optimization.OptimizerGeneric(problem)
        optimizer.optimize(maxiter=10, disp=False, tol=1e-3)
        optimizer.undo()
        assert len(optimizer._x) == 0

    def test_fun_nan_rss(self):
        """
        Tests that a NaN value from an operand is handled gracefully and
        results in a large penalty value.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lens = UVReflectingMicroscope()
            lens.set_radius(0.2, 3)  # This breaks the lens, causing NaNs
            problem = optimization.OptimizationProblem(optic=lens)
            problem.add_operand("rms_spot_size", target=0.0, Hx=0, Hy=0.1, num_rays=100, surface_number=-1)
            optimizer = optimization.OptimizerGeneric(problem)
            assert optimizer._fun(be.array([0.2])) == 1e10


class TestLeastSquares:
    """
    Tests the LeastSquares optimizer, which uses SciPy's least_squares.
    """

    def test_optimize(self):
        """Tests a simple optimization with bounds."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("conic", 1, min_val=-1, max_val=1)
        problem.add_operand("f2", target=90)
        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(maxiter=10, disp=False, tol=1e-3)
        assert result.success

    def test_no_bounds(self):
        """Tests optimization without variable bounds."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("conic", 1)
        problem.add_operand("f2", target=90)
        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(maxiter=10, disp=False, tol=1e-3)
        assert result.success

    def test_verbose(self, capsys):
        """Tests that verbose output is printed correctly."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1, min_val=-1000)
        problem.add_operand("f2", target=90)
        optimizer = optimization.LeastSquares(problem)
        optimizer.optimize(maxiter=1, disp=True)
        captured = capsys.readouterr()
        assert "Cost" in captured.out

    def test_method_trf_with_bounds(self):
        """Tests the 'trf' method with bounded variables."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1, min_val=10, max_val=100)
        problem.add_operand("f2", target=90)
        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(method_choice="trf", maxiter=10)
        assert result.success

    def test_method_dogbox_with_bounds(self):
        """Tests the 'dogbox' method with bounded variables."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1, min_val=10, max_val=100)
        problem.add_operand("f2", target=90)
        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(method_choice="dogbox", maxiter=10)
        assert result.success

    def test_method_lm_with_bounds_warning(self, capsys):
        """
        Tests that using the 'lm' method with bounds raises a warning, as it
        does not support them.
        """
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1, min_val=10, max_val=100)
        problem.add_operand("f2", target=90)
        optimizer = optimization.LeastSquares(problem)
        optimizer.optimize(method_choice="lm", maxiter=5)
        captured = capsys.readouterr()
        assert "Warning: Method 'lm' (Levenberg-Marquardt) chosen" in captured.out

    def test_unknown_method_choice_warning(self, capsys):
        """Tests that an unknown method choice defaults to 'trf' with a warning."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1)
        problem.add_operand("f2", target=90)
        optimizer = optimization.LeastSquares(problem)
        optimizer.optimize(method_choice="unknown_method", maxiter=5)
        captured = capsys.readouterr()
        assert "Warning: Unknown method_choice 'unknown_method'" in captured.out


class TestLeastSquaresErrorHandling:
    """
    Tests the error handling capabilities of the LeastSquares optimizer,
    particularly for NaN values and exceptions from operands.
    """

    def test_nan_residual_handling(self):
        """
        Tests that if an operand returns NaN, it is replaced with a large
        penalty value.
        """
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1, min_val=10, max_val=100)
        problem.operands.operands.append(MockOperandNaN())
        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(maxiter=5)
        assert be.isclose(result.cost, 0.5 * 1e10)

    def test_exception_in_residual_handling(self):
        """
        Tests that if an operand raises an exception, it is caught and
        replaced with a large penalty value.
        """
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1, min_val=10, max_val=100)
        problem.operands.operands.append(MockOperandException())
        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(maxiter=5)
        assert be.isclose(result.cost, 0.5 * 1e10)

    def test_optimize_no_operands(self):
        """
        Tests that optimization with no operands completes successfully with
        zero cost.
        """
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1, min_val=10, max_val=100)
        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(maxiter=5)
        assert result.success
        assert be.isclose(result.cost, 0.0)


class TestDualAnnealing:
    """Tests the DualAnnealing global optimizer."""

    def test_optimize(self):
        """Tests a simple optimization run."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("thickness", 1, min_val=10, max_val=100)
        problem.add_operand("f2", target=95)
        optimizer = optimization.DualAnnealing(problem)
        result = optimizer.optimize(maxiter=10, disp=False)
        assert result.success

    def test_raise_error_no_bounds(self):
        """Tests that running without bounds raises a ValueError."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("thickness", 1)
        problem.add_operand("f2", target=95)
        optimizer = optimization.DualAnnealing(problem)
        with pytest.raises(ValueError):
            optimizer.optimize(maxiter=10)


class TestDifferentialEvolution:
    """Tests the DifferentialEvolution global optimizer."""

    def test_optimize(self):
        """Tests a simple optimization run."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("index", 1, min_val=1.2, max_val=1.8, wavelength=0.5)
        problem.add_operand("f2", target=90)
        optimizer = optimization.DifferentialEvolution(problem)
        result = optimizer.optimize(maxiter=10, disp=False, workers=1)
        assert result.success

    def test_raise_error_no_bounds(self):
        """Tests that running without bounds raises a ValueError."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("index", 1, wavelength=0.5)
        problem.add_operand("f2", target=95)
        optimizer = optimization.DifferentialEvolution(problem)
        with pytest.raises(ValueError):
            optimizer.optimize(maxiter=10)

    def test_workers(self):
        """Tests that the optimizer runs with parallel workers."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("index", 1, min_val=1.2, max_val=1.8, wavelength=0.5)
        problem.add_operand("f2", target=90)
        optimizer = optimization.DifferentialEvolution(problem)
        result = optimizer.optimize(maxiter=10, disp=False, workers=-1)
        assert result.success


class TestSHGO:
    """Tests the SHGO (Simplicial Homology Global Optimization) optimizer."""

    def test_optimize(self):
        """Tests a simple optimization run."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1, min_val=10, max_val=100)
        problem.add_operand("f2", target=90)
        optimizer = optimization.SHGO(problem)
        result = optimizer.optimize()
        assert result.success

    def test_raise_error_no_bounds(self):
        """Tests that running without bounds raises a ValueError."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1)
        problem.add_operand("f2", target=90)
        optimizer = optimization.SHGO(problem)
        with pytest.raises(ValueError):
            optimizer.optimize()


class TestBasinHopping:
    """Tests the BasinHopping global optimizer."""

    def test_optimize(self):
        """Tests a simple optimization run."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1)
        problem.add_operand("f2", target=90)
        optimizer = optimization.BasinHopping(problem)
        result = optimizer.optimize(niter=10)
        assert result.fun < problem.initial_value

    def test_raise_error_with_bounds(self):
        """Tests that running with bounds raises a ValueError."""
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1, min_val=10, max_val=100)
        problem.add_operand("f2", target=90)
        optimizer = optimization.BasinHopping(problem)
        with pytest.raises(ValueError):
            optimizer.optimize(niter=10)


class TestGlassExpert:
    """Tests the GlassExpert, a specialized routine for glass selection."""

    def test_optimize(self):
        """Tests a standard glass optimization run."""
        problem = optimization.OptimizationProblem(optic=CookeTriplet())
        problem.add_variable("thickness", 1, min_val=10, max_val=100)
        problem.add_variable("material", 1, glass_selection=["N-BK7", "N-SK2"])
        problem.add_operand("f2", target=95)
        optimizer = glass_expert.GlassExpert(problem)
        result = optimizer.run(num_neighbours=1, maxiter=8, disp=False, verbose=False)
        assert result.success

    def test_optimize_fail_num_neighbours(self):
        """
        Tests that an assertion is raised if `num_neighbours` is greater
        than the number of available glasses.
        """
        problem = optimization.OptimizationProblem(optic=CookeTriplet())
        problem.add_variable("material", 1, glass_selection=["N-BK7"])
        problem.add_operand("f2", target=95)
        optimizer = glass_expert.GlassExpert(problem)
        with pytest.raises(AssertionError):
            optimizer.run(num_neighbours=2, maxiter=8)

    def test_vprint(self, capsys):
        """Tests the verbose printing utility."""
        problem = optimization.OptimizationProblem(optic=CookeTriplet())
        optimizer = glass_expert.GlassExpert(problem)
        optimizer.verbose = True
        optimizer.vprint("Hello World")
        captured = capsys.readouterr()
        assert "Hello World" in captured.out
        optimizer.verbose = False
        optimizer.vprint("This should not appear")
        captured = capsys.readouterr()
        assert captured.out == ""


class TestOptimizerWithBounds:
    """
    Tests optimizers with more complex bound scenarios, particularly those
    involving variable scalers.
    """

    def test_optimize_with_reciprocal_scaler_and_min_bounds(self):
        """Tests optimization with a reciprocal scaler and a minimum bound."""
        from optiland.optimization.scaling.reciprocal import ReciprocalScaler
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1, scaler=ReciprocalScaler(), min_val=500.0)
        problem.add_operand("f2", target=90)
        optimizer = optimization.OptimizerGeneric(problem)
        result = optimizer.optimize(maxiter=10, disp=False)
        assert result.success
        assert problem.optic.surface_group.surfaces[1].geometry.radius >= 500.0

    def test_optimize_with_reciprocal_scaler_and_max_bounds(self):
        """Tests optimization with a reciprocal scaler and a maximum bound."""
        from optiland.optimization.scaling.reciprocal import ReciprocalScaler
        lens = Microscope20x()
        lens.set_radius(-1000, 1)
        problem = optimization.OptimizationProblem(optic=lens)
        problem.add_variable("radius", 1, scaler=ReciprocalScaler(), max_val=-500.0)
        problem.add_operand("f2", target=90)
        optimizer = optimization.OptimizerGeneric(problem)
        result = optimizer.optimize(maxiter=10, disp=False)
        assert result.success
        assert problem.optic.surface_group.surfaces[1].geometry.radius <= -500.0

    def test_optimize_with_reciprocal_scaler_and_crossing_bounds(self):
        """
        Tests that a ValueError is raised if bounds cross zero when using a
        reciprocal scaler.
        """
        from optiland.optimization.scaling.reciprocal import ReciprocalScaler
        problem = optimization.OptimizationProblem(optic=Microscope20x())
        problem.add_variable("radius", 1, scaler=ReciprocalScaler(), min_val=-100, max_val=100)
        with pytest.raises(ValueError):
            _ = problem.variables[0].bounds