import warnings

import optiland.backend as be
import pytest

from optiland.optimization import optimization
from optiland.samples.microscopes import (
    Microscope20x,
    Objective60x,
    UVReflectingMicroscope,
)


class TestOptimizationProblem:
    def test_add_operand(self):
        lens = Objective60x()
        problem = optimization.OptimizationProblem()
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=50,
            weight=1.0,
            input_data=input_data,
        )
        assert len(problem.operands) == 1
        assert problem.operands[0].operand_type == "f2"
        assert problem.operands[0].target == 50
        assert problem.operands[0].weight == 1.0
        assert problem.operands[0].input_data == input_data

    def test_add_variable(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        assert len(problem.variables) == 1
        assert problem.variables[0].type == "radius"
        assert problem.variables[0].min_val == 10
        assert problem.variables[0].max_val == 100
        assert problem.variables[0].surface_number == 1

    def test_clear_operands(self):
        lens = Objective60x()
        problem = optimization.OptimizationProblem()
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=50,
            weight=1.0,
            input_data=input_data,
        )
        problem.clear_operands()
        assert len(problem.operands) == 0

    def test_clear_variables(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        problem.clear_variables()
        assert len(problem.variables) == 0

    def test_fun_array(self):
        lens = Objective60x()
        problem = optimization.OptimizationProblem()
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=98.57864671748113,
            weight=1.0,
            input_data=input_data,
        )
        fun_array = problem.fun_array()
        assert be.allclose(fun_array, be.array([0.0]))

    def test_sum_squared(self):
        lens = Objective60x()
        problem = optimization.OptimizationProblem()
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        sum_squared = problem.sum_squared()
        val = (lens.paraxial.f2() - 90) ** 2
        assert be.isclose(sum_squared, val)

    def test_rss(self):
        lens = UVReflectingMicroscope()
        problem = optimization.OptimizationProblem()
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        rss = problem.rss()
        val = be.abs(lens.paraxial.f2() - 90)
        assert be.isclose(rss, val)

    def test_update_optics(self):
        lens = UVReflectingMicroscope()
        problem = optimization.OptimizationProblem()
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        problem.update_optics()
        # No assertion needed, just ensure no exceptions are raised

    def test_operand_info(self, capsys):
        lens = UVReflectingMicroscope()
        problem = optimization.OptimizationProblem()
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=9090,
            weight=1.0,
            input_data=input_data,
        )
        problem.operand_info()
        captured = capsys.readouterr()
        assert "Operand Type" in captured.out
        assert "Target" in captured.out
        assert "Min. Bound" in captured.out
        assert "Max. Bound" in captured.out
        assert "Weight" in captured.out
        assert "Value" in captured.out
        assert "Delta" in captured.out
        assert "Contrib. [%]" in captured.out

    def test_variable_info(self, capsys):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        problem.variable_info()
        captured = capsys.readouterr()
        assert "Variable Type" in captured.out
        assert "Surface" in captured.out
        assert "Value" in captured.out
        assert "Min. Bound" in captured.out
        assert "Max. Bound" in captured.out

    def test_merit_info(self, capsys):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        problem.merit_info()
        captured = capsys.readouterr()
        assert "Merit Function Value" in captured.out
        assert "Improvement (%)" in captured.out

        # case when initial merit value != 0.0
        problem.initial_value = 10.0
        problem.merit_info()
        captured = capsys.readouterr()
        assert "Merit Function Value" in captured.out
        assert "Improvement (%)" in captured.out

    def test_info(self, capsys):
        lens = UVReflectingMicroscope()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        problem.info()
        captured = capsys.readouterr()
        assert "Merit Function Value" in captured.out
        assert "Operand Type" in captured.out
        assert "Variable Type" in captured.out


class TestOptimizerGeneric:
    def test_optimize(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.OptimizerGeneric(problem)
        result = optimizer.optimize(maxiter=10, disp=False, tol=1e-3)
        assert result.success

    def test_undo(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.OptimizerGeneric(problem)
        optimizer.optimize(maxiter=10, disp=False, tol=1e-3)
        optimizer.undo()
        assert len(optimizer._x) == 0

    def test_fun_nan_rss(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            lens = UVReflectingMicroscope()
            # this will "break" the lens, resulting in NaN (for testing)
            lens.set_radius(0.2, 3)
            problem = optimization.OptimizationProblem()
            input_data = {
                "optic": lens,
                "Hx": 0.0,
                "Hy": 0.1,
                "wavelength": 0.5,
                "num_rays": 100,
                "surface_number": -1,
            }
            problem.add_operand(
                operand_type="rms_spot_size",
                target=0.0,
                weight=1.0,
                input_data=input_data,
            )
            optimizer = optimization.OptimizerGeneric(problem)
            assert optimizer._fun(be.array([0.2])) == 1e10


class TestLeastSquares:
    def test_optimize(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "conic", surface_number=1, min_val=-1, max_val=1)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(maxiter=10, disp=False, tol=1e-3)
        assert result.success

    def test_no_bounds(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "conic", surface_number=1)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(maxiter=10, disp=False, tol=1e-3)
        assert result.success

    def test_verbose(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(
            lens,
            "radius",
            surface_number=1,
            min_val=-1000,
            max_val=None,
        )
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(maxiter=100, disp=True, tol=1e-3)
        assert result.success

    def test_method_trf_with_bounds(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        min_b, max_b = 10, 100
        problem.add_variable(
            lens, "radius", surface_number=1, min_val=min_b, max_val=max_b
        )
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(method_choice="trf", maxiter=10, tol=1e-3)
        assert result.success
        # Check if the optimized variable is within bounds (SciPy's TRF handles this)
        optimized_radius = lens.surface_group.surfaces[1].geometry.radius
        assert min_b <= optimized_radius <= max_b

    def test_method_dogbox_with_bounds(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        min_b, max_b = 10, 100
        problem.add_variable(
            lens, "radius", surface_number=1, min_val=min_b, max_val=max_b
        )
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(method_choice="dogbox", maxiter=10, tol=1e-3)
        assert result.success
        optimized_radius = lens.surface_group.surfaces[1].geometry.radius
        assert min_b <= optimized_radius <= max_b

    def test_method_lm_with_bounds_warning(self, capsys):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.LeastSquares(problem)
        optimizer.optimize(method_choice="lm", maxiter=5)
        captured = capsys.readouterr()
        expected_warning = (
            "Warning: Method 'lm' (Levenberg-Marquardt) chosen, "
            "but variable bounds are set. SciPy's 'lm' method does not "
            "support bounds; bounds will be ignored."
        )
        assert expected_warning in captured.out

    def test_unknown_method_choice_warning(self, capsys):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1)  # No bounds needed
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.LeastSquares(problem)
        optimizer.optimize(method_choice="unknown_method", maxiter=5)
        captured = capsys.readouterr()
        expected_warning = (
            "Warning: Unknown method_choice 'unknown_method'. Defaulting to "
            "'trf' method."  # Updated expected warning
        )
        assert expected_warning in captured.out


class MockOperandNaN:
    def __init__(self, target=0, weight=1):
        self.target = target
        self.weight = weight
        self.operand_type = "mock_nan"
        self.input_data = {}

    def fun(self):
        return be.nan

    def delta(self):
        return be.nan

    def value(self):  # Add a value method
        return be.nan


class MockOperandException:
    def __init__(self, target=0, weight=1):
        self.target = target
        self.weight = weight
        self.operand_type = "mock_exception"
        self.input_data = {}

    def fun(self):
        raise RuntimeError("Test Exception from mock operand")

    def delta(self):
        raise RuntimeError("Test Exception from mock operand")

    def value(self):
        raise RuntimeError("Test Exception from mock operand")


class TestLeastSquaresErrorHandling:
    def test_nan_residual_handling(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        # Add a dummy variable, its properties don't matter much here
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)

        mock_op = MockOperandNaN()
        problem.initial_value = (
            1.0  # Set to non-zero to prevent sum_squared in OptimizerGeneric init
        )
        problem.operands.operands.append(mock_op)  # Manually add mock operand

        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(maxiter=5)

        # Cost is 0.5 * sum(residuals**2). Residual is sqrt(1e10 / 1) = sqrt(1e10)
        # So cost = 0.5 * (sqrt(1e10))^2 = 0.5 * 1e10
        assert be.isclose(result.cost, 0.5 * 1e10)
        # Check that optimization completed without crashing (status might vary)
        assert result.status is not None  # General check for completion

    def test_exception_in_residual_handling(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)

        mock_op = MockOperandException()
        problem.initial_value = (
            1.0  # Set to non-zero to prevent sum_squared in OptimizerGeneric init
        )
        problem.operands.operands.append(mock_op)

        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(maxiter=5)
        assert be.isclose(result.cost, 0.5 * 1e10)
        assert result.status is not None

    def test_optimize_no_operands(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        # No operands are added

        optimizer = optimization.LeastSquares(problem)
        result = optimizer.optimize(maxiter=5)

        assert result.success  # Or a similar status indicating valid completion
        assert be.isclose(result.cost, 0.0)
        assert len(result.fun) == 0


class TestDualAnnealing:
    def test_optimize(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(
            lens,
            "thickness",
            surface_number=1,
            min_val=10,
            max_val=100,
        )
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=95,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.DualAnnealing(problem)
        result = optimizer.optimize(maxiter=10, disp=False)
        assert result.success

    def test_raise_error_no_bounds(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "thickness", surface_number=1)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=95,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.DualAnnealing(problem)
        with pytest.raises(ValueError):
            optimizer.optimize(maxiter=10, disp=False)


class TestDifferentialEvolution:
    def test_optimize(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(
            lens,
            "index",
            surface_number=1,
            min_val=1.2,
            max_val=1.8,
            wavelength=0.5,
        )
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.DifferentialEvolution(problem)
        result = optimizer.optimize(maxiter=10, disp=False, workers=1)
        assert result.success

    def test_raise_error_no_bounds(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "index", surface_number=1, wavelength=0.5)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=95,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.DifferentialEvolution(problem)
        with pytest.raises(ValueError):
            optimizer.optimize(maxiter=10, disp=False)

    def test_workers(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(
            lens,
            "index",
            surface_number=1,
            min_val=1.2,
            max_val=1.8,
            wavelength=0.5,
        )
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.DifferentialEvolution(problem)
        result = optimizer.optimize(maxiter=10, disp=False, workers=1)
        assert result.success


class TestSHGO:
    def test_optimize(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.SHGO(problem)
        result = optimizer.optimize()
        assert result.success

    def test_raise_error_no_bounds(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.SHGO(problem)
        with pytest.raises(ValueError):
            optimizer.optimize()


class TestBasinHopping:
    def test_optimize(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.BasinHopping(problem)
        result = optimizer.optimize(niter=10)
        assert result.fun < problem.initial_value

    def test_raise_error_with_bounds(self):
        lens = Microscope20x()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1, min_val=10, max_val=100)
        input_data = {"optic": lens}
        problem.add_operand(
            operand_type="f2",
            target=90,
            weight=1.0,
            input_data=input_data,
        )
        optimizer = optimization.BasinHopping(problem)
        with pytest.raises(ValueError):
            optimizer.optimize(niter=10)
