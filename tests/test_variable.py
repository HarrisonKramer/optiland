import optiland.backend as be
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import (
    ChebyshevPolynomialGeometry,
    PolynomialGeometry,
    ZernikePolynomialGeometry,
)
from optiland.optimization import variable, OptimizationProblem, OptimizerGeneric
from optiland.samples.microscopes import Objective60x, UVReflectingMicroscope
from optiland.samples.simple import AsphericSinglet, Edmund_49_847


class TestRadiusVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.radius_var = variable.RadiusVariable(self.optic, 1)

    def test_get_value(self):
        assert be.isclose(self.radius_var.get_value(), 4.5325999999999995)

    def test_update_value(self):
        self.radius_var.update_value(5.0)
        assert be.isclose(self.radius_var.get_value(), 5.0)


class TestReciprocalRadiusVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Edmund_49_847()
        self.reciprocal_radius_var = variable.ReciprocalRadiusVariable(self.optic, 1)
        self.radius_var = variable.RadiusVariable(self.optic, 1, apply_scaling=False)
        self.scaling = 10.0
        self.problem = OptimizationProblem()
        input_data = {
            "optic": self.optic,
            "Hx": 0.0,
            "Hy": 0.0,
            "num_rays": 5,
            "wavelength": self.optic.wavelengths.primary_wavelength.value,
            "distribution": "hexapolar",
            "surface_number": 3,
        }
        self.problem.add_operand(
            operand_type="rms_spot_size", target=0, weight=1, input_data=input_data
        )

    def test_get_value(self):
        # Expected reciprocal = 1 / radius; radius from TestRadiusVariable ≈ 553.260
        expected = self.scaling * (1.0 / self.radius_var.get_value())
        assert be.isclose(self.reciprocal_radius_var.get_value(), expected, atol=1e-4)
        a = str(self.reciprocal_radius_var)
        assert a.startswith(
            "Reciprocal Radius of Curvature - Surface 1 - scaled value: 0.50"
        )

    def test_get_value_without_scaling(self):
        self.reciprocal_radius_var = variable.ReciprocalRadiusVariable(
            self.optic, 1, apply_scaling=False
        )
        expected = 1.0 / self.radius_var.get_value()
        assert be.isclose(self.reciprocal_radius_var.get_value(), expected, atol=1e-4)
        a = str(self.reciprocal_radius_var)
        assert a.startswith(
            "Reciprocal Radius of Curvature - Surface 1 - unscaled value: 0.050"
        )

    def test_update_value(self):
        # Update reciprocal value to 0.25, expect new radius = 1/0.25 = 4.0
        self.reciprocal_radius_var.update_value(0.25)
        expected_radius = self.scaling * (1.0 / 0.25)
        assert be.isclose(self.radius_var.get_value(), expected_radius, atol=1e-4)
        assert be.isclose(self.reciprocal_radius_var.get_value(), 0.25, atol=1e-4)

    def test_get_value_infinity(self):
        # Set the surface radius to 0 so reciprocal becomes infinity (division by zero)
        self.radius_var.update_value(0.0)
        val = self.reciprocal_radius_var.get_value()
        assert val == be.inf

    def test_update_value_zero(self):
        # Update reciprocal value to 0 -> new radius set to infinity, so reciprocal becomes 0
        self.reciprocal_radius_var.update_value(0.0)
        assert be.isclose(self.reciprocal_radius_var.get_value(), 0.0)

    def test_optimization(self):
        # Add the reciprocal radius variable for the first surface
        self.problem.add_variable(self.optic, "reciprocal_radius", surface_number=1)

        # Run the optimization
        optimizer = OptimizerGeneric(self.problem)

        # this will set the radius of surface 1 to scale*(1/0.1) = 10*(10) = 100
        self.reciprocal_radius_var.update_value(0.1)
        optimizer.optimize(tol=1e-9)

        # Check if the radius of the first surface is close to the expected value
        expected_radius = 19.93  # Expected value from the initial definition
        optimized_radius = self.radius_var.get_value()
        # just make sure the final value is in the ballpark, and there were no exceptions thrown
        assert be.isclose(optimized_radius, expected_radius, atol=5)

    def test_optimization_with_flat_surface(self):
        # Add the reciprocal radius variable for the first surface
        self.problem.add_variable(self.optic, "reciprocal_radius", surface_number=1)

        # Make the first surface mathematically flat
        self.radius_var.update_value(-be.inf)

        # Run the optimization
        optimizer = OptimizerGeneric(self.problem)
        optimizer.optimize(tol=1e-9)

        # the optimization above stops prematurely, so we have to try again.
        # TODO: understand the problem and fix it or at least properly document it.
        # trying again:
        optimizer.optimize(tol=1e-9)

        # Check if the radius of the first surface is close to the expected value
        expected_radius = 19.93  # Expected value from the initial definition
        optimized_radius = self.radius_var.get_value()
        # just make sure the final value is in the ballpark, and there were no exceptions thrown
        assert be.isclose(optimized_radius, expected_radius, atol=5)


class TestConicVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = UVReflectingMicroscope()
        self.conic_var = variable.ConicVariable(self.optic, 1)

    def test_get_value(self):
        assert be.isclose(self.conic_var.get_value(), 0.0)

    def test_update_value(self):
        self.conic_var.update_value(-0.5)
        assert be.isclose(self.conic_var.get_value(), -0.5)

    def test_scale_value(self):
        self.conic_var.scale(2.0)
        assert be.isclose(self.conic_var.get_value(), 0.0)

    def test_inverse_scale_value(self):
        self.conic_var.inverse_scale(2.0)
        assert be.isclose(self.conic_var.get_value(), 0.0)

    def test_string_representation(self):
        assert str(self.conic_var) == "Conic Constant, Surface 1"


class TestThicknessVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.thickness_var = variable.ThicknessVariable(self.optic, 2)

    def test_get_value(self):
        assert be.isclose(self.thickness_var.get_value(), -0.5599999999999994)

    def test_update_value(self):
        self.thickness_var.update_value(-0.6)
        assert be.isclose(self.thickness_var.get_value(), -0.6)

    def test_get_value_no_scaling(self):
        self.optic = Objective60x()
        self.thickness_var = variable.ThicknessVariable(
            self.optic,
            2,
            apply_scaling=False,
        )
        assert be.isclose(self.thickness_var.get_value(), 4.4)


class TestIndexVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.index_var = variable.IndexVariable(self.optic, 1, 0.55)

    def test_get_value(self):
        assert be.isclose(self.index_var.get_value(), -0.012206444700957775)

    def test_update_value(self):
        self.index_var.update_value(0.1)
        assert be.isclose(self.index_var.get_value(), 0.1)

    def test_get_value_no_scaling(self):
        self.optic = Objective60x()
        self.index_var = variable.IndexVariable(
            self.optic,
            1,
            0.55,
            apply_scaling=False,
        )
        assert be.isclose(self.index_var.get_value(), 1.4877935552990422)

    def test_string_representation(self):
        assert str(self.index_var) == "Refractive Index, Surface 1"


class TestAsphereCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        self.asphere_var = variable.AsphereCoeffVariable(self.optic, 1, 0)

    def test_get_value(self):
        assert be.isclose(self.asphere_var.get_value(), -2.248851)

    def test_update_value(self):
        self.asphere_var.update_value(-2.0)
        assert be.isclose(self.asphere_var.get_value(), -2.0)

    def test_get_value_no_scaling(self):
        self.optic = AsphericSinglet()
        self.asphere_var = variable.AsphereCoeffVariable(
            self.optic,
            1,
            0,
            apply_scaling=False,
        )
        assert be.isclose(self.asphere_var.get_value(), -0.0002248851)

    def test_string_representation(self):
        assert str(self.asphere_var) == "Asphere Coeff. 0, Surface 1"


class TestPolynomialCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        poly_geo = PolynomialGeometry(
            CoordinateSystem(),
            100,
            coefficients=be.zeros((3, 3)),
        )
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.PolynomialCoeffVariable(self.optic, 0, (1, 1))

    def test_get_value(self):
        assert self.poly_var.get_value() == 0.0

    def test_update_value(self):
        self.poly_var.update_value(1.0)
        assert be.isclose(self.poly_var.get_value(), 1.0)

    def test_get_value_index_error(self):
        self.optic = AsphericSinglet()
        poly_geo = PolynomialGeometry(CoordinateSystem(), 100)
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.PolynomialCoeffVariable(self.optic, 0, (1, 1))
        assert self.poly_var.get_value() == 0.0

    def test_update_value_index_error(self):
        self.optic = AsphericSinglet()
        poly_geo = PolynomialGeometry(CoordinateSystem(), 100)
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.PolynomialCoeffVariable(self.optic, 0, (1, 1))
        self.poly_var.update_value(1.0)
        assert be.isclose(self.poly_var.get_value(), 1.0)

    def test_string_representation(self):
        assert str(self.poly_var) == "Poly. Coeff. (1, 1), Surface 0"

    def test_get_value_no_scaling(self):
        self.optic = AsphericSinglet()
        poly_geo = PolynomialGeometry(
            CoordinateSystem(),
            100,
            coefficients=be.zeros((3, 3)),
        )
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.PolynomialCoeffVariable(
            self.optic,
            0,
            (1, 1),
            apply_scaling=False,
        )
        assert self.poly_var.get_value() == 0.0


class TestChebyshevCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        poly_geo = ChebyshevPolynomialGeometry(
            CoordinateSystem(),
            100,
            coefficients=be.zeros((3, 3)),
        )
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ChebyshevCoeffVariable(self.optic, 0, (1, 1))

    def test_get_value(self):
        assert self.poly_var.get_value() == 0.0

    def test_update_value(self):
        self.poly_var.update_value(1.0)
        assert be.isclose(self.poly_var.get_value(), 1.0)

    def test_get_value_index_error(self):
        self.optic = AsphericSinglet()
        poly_geo = ChebyshevPolynomialGeometry(CoordinateSystem(), 100)
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ChebyshevCoeffVariable(self.optic, 0, (1, 1))
        assert self.poly_var.get_value() == 0.0

    def test_update_value_index_error(self):
        self.optic = AsphericSinglet()
        poly_geo = ChebyshevPolynomialGeometry(CoordinateSystem(), 100)
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ChebyshevCoeffVariable(self.optic, 0, (1, 1))
        self.poly_var.update_value(1.0)
        assert be.isclose(self.poly_var.get_value(), 1.0)

    def test_string_representation(self):
        assert str(self.poly_var) == "Chebyshev Coeff. (1, 1), Surface 0"


class TestZernikeCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        poly_geo = ZernikePolynomialGeometry(
            CoordinateSystem(),
            100,
            coefficients=be.zeros(3),
        )
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ZernikeCoeffVariable(self.optic, 0, 1)

    def test_get_value(self):
        assert self.poly_var.get_value() == 0.0

    def test_update_value(self):
        self.poly_var.update_value(1.0)
        assert be.isclose(self.poly_var.get_value(), 1.0)

    def test_get_value_index_error(self):
        self.optic = AsphericSinglet()
        poly_geo = ZernikePolynomialGeometry(CoordinateSystem(), 100)
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ZernikeCoeffVariable(self.optic, 0, 1)
        assert self.poly_var.get_value() == 0.0

    def test_update_value_index_error(self):
        self.optic = AsphericSinglet()
        poly_geo = ZernikePolynomialGeometry(CoordinateSystem(), 100)
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ZernikeCoeffVariable(self.optic, 0, 1)
        self.poly_var.update_value(1.0)
        assert be.isclose(self.poly_var.get_value(), 1.0)

    def test_string_representation(self):
        assert str(self.poly_var) == "Zernike Coeff. 1, Surface 0"


class TestVariable:
    def test_get_value(self):
        optic = Objective60x()
        radius_var = variable.Variable(optic, "radius", surface_number=1)
        assert be.isclose(radius_var.value, 4.5325999999999995)

    def test_unrecognized_attribute(self):
        optic = Objective60x()
        radius_var = variable.Variable(
            optic,
            "radius",
            surface_number=1,
            unrecognized_attribute=1,
        )
        assert be.isclose(radius_var.value, 4.5325999999999995)

    def test_invalid_type(self):
        optic = Objective60x()
        with pytest.raises(ValueError):
            variable.Variable(optic, "invalid", surface_number=1)


class TestTiltVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.tilt_var_x = variable.TiltVariable(self.optic, 1, "x")
        self.tilt_var_y = variable.TiltVariable(self.optic, 1, "y")

    def test_get_value_x(self):
        assert be.isclose(self.tilt_var_x.get_value(), 0.0)

    def test_get_value_y(self):
        assert be.isclose(self.tilt_var_y.get_value(), 0.0)

    def test_update_value_x(self):
        self.tilt_var_x.update_value(5.0)
        assert be.isclose(self.tilt_var_x.get_value(), 5.0)

    def test_update_value_y(self):
        self.tilt_var_y.update_value(5.0)
        assert be.isclose(self.tilt_var_y.get_value(), 5.0)

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            variable.TiltVariable(self.optic, 1, "z")

    def test_str(self):
        assert str(self.tilt_var_x) == "Tilt X, Surface 1"
        assert str(self.tilt_var_y) == "Tilt Y, Surface 1"

    def test_get_value_no_scaling(self):
        self.optic = Objective60x()
        self.tilt_var_x = variable.TiltVariable(self.optic, 1, "x", apply_scaling=False)
        assert be.isclose(self.tilt_var_x.get_value(), 0.0)


class TestDecenterVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.decenter_var_x = variable.DecenterVariable(self.optic, 1, "x")
        self.decenter_var_y = variable.DecenterVariable(self.optic, 1, "y")

    def test_get_value_x(self):
        assert be.isclose(self.decenter_var_x.get_value(), 0.0)

    def test_get_value_y(self):
        assert be.isclose(self.decenter_var_y.get_value(), 0.0)

    def test_update_value_x(self):
        self.decenter_var_x.update_value(5.0)
        assert be.isclose(self.decenter_var_x.get_value(), 5.0)

    def test_update_value_y(self):
        self.decenter_var_y.update_value(5.0)
        assert be.isclose(self.decenter_var_y.get_value(), 5.0)

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            variable.DecenterVariable(self.optic, 1, "z")

    def test_str(self):
        assert str(self.decenter_var_x) == "Decenter X, Surface 1"
        assert str(self.decenter_var_y) == "Decenter Y, Surface 1"

    def test_get_value_no_scaling(self):
        self.optic = Objective60x()
        self.decenter_var_x = variable.DecenterVariable(
            self.optic,
            1,
            "x",
            apply_scaling=False,
        )
        assert be.isclose(self.decenter_var_x.get_value(), 0.0)


class TestVariableManager:
    def test_add(self):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        assert len(var_manager) == 1

    def test_clear(self):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        var_manager.clear()
        assert len(var_manager) == 0

    def test_iter(self):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        var_manager.add(optic, "radius", surface_number=2)
        for var in var_manager:
            assert isinstance(var, variable.Variable)

    def test_getitem(self):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        assert isinstance(var_manager[0], variable.Variable)

    def test_setitem(self):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        var_manager[0] = variable.Variable(optic, "radius", surface_number=2)
        assert isinstance(var_manager[0], variable.Variable)
        assert var_manager[0].surface_number == 2
        assert len(var_manager) == 1

    def test_len(self):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        var_manager.add(optic, "radius", surface_number=2)
        assert len(var_manager) == 2

    def test_getitem_index_error(self):
        var_manager = variable.VariableManager()
        with pytest.raises(IndexError):
            var_manager[0]

    def test_setitem_index_error(self):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        with pytest.raises(IndexError):
            var_manager[0] = variable.Variable(optic, "radius", surface_number=2)

    def test_setitem_invalid_type(self):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        with pytest.raises(ValueError):
            var_manager[0] = "invalid"

    def test_iterable(self):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        var_manager.add(optic, "radius", surface_number=2)
        for i, var in enumerate(var_manager):
            assert isinstance(var, variable.Variable)
            assert var.surface_number == i + 1
        assert i == 1

    def test_delitem(self):
        optic = Objective60x()
        var_manager = variable.VariableManager()
        var_manager.add(optic, "radius", surface_number=1)
        del var_manager[0]
        assert len(var_manager) == 0
