import pytest
import numpy as np
from optiland.optimization import variable
from optiland.geometries import PolynomialGeometry, ChebyshevPolynomialGeometry
from optiland.coordinate_system import CoordinateSystem
from optiland.samples.microscopes import Objective60x, UVReflectingMicroscope
from optiland.samples.simple import AsphericSinglet


class TestRadiusVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.radius_var = variable.RadiusVariable(self.optic, 1)

    def test_get_value(self):
        assert np.isclose(self.radius_var.get_value(), 4.5325999999999995)

    def test_update_value(self):
        self.radius_var.update_value(5.0)
        assert np.isclose(self.radius_var.get_value(), 5.0)


class TestConicVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = UVReflectingMicroscope()
        self.conic_var = variable.ConicVariable(self.optic, 1)

    def test_get_value(self):
        assert np.isclose(self.conic_var.get_value(), 0.0)

    def test_update_value(self):
        self.conic_var.update_value(-0.5)
        assert np.isclose(self.conic_var.get_value(), -0.5)

    def test_scale_value(self):
        self.conic_var.scale(2.0)
        assert np.isclose(self.conic_var.get_value(), 0.0)

    def test_inverse_scale_value(self):
        self.conic_var.inverse_scale(2.0)
        assert np.isclose(self.conic_var.get_value(), 0.0)

    def test_string_representation(self):
        assert str(self.conic_var) == 'Conic Constant, Surface 1'


class TestThicknessVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.thickness_var = variable.ThicknessVariable(self.optic, 2)

    def test_get_value(self):
        assert np.isclose(self.thickness_var.get_value(), -0.5599999999999994)

    def test_update_value(self):
        self.thickness_var.update_value(-0.6)
        assert np.isclose(self.thickness_var.get_value(), -0.6)

    def test_get_value_no_scaling(self):
        self.optic = Objective60x()
        self.thickness_var = variable.ThicknessVariable(self.optic, 2,
                                                        apply_scaling=False)
        assert np.isclose(self.thickness_var.get_value(), 4.4)


class TestIndexVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.index_var = variable.IndexVariable(self.optic, 1, 0.55)

    def test_get_value(self):
        assert np.isclose(self.index_var.get_value(), -0.012206444700957775)

    def test_update_value(self):
        self.index_var.update_value(0.1)
        assert np.isclose(self.index_var.get_value(), 0.1)

    def test_get_value_no_scaling(self):
        self.optic = Objective60x()
        self.index_var = variable.IndexVariable(self.optic, 1, 0.55,
                                                apply_scaling=False)
        assert np.isclose(self.index_var.get_value(), 1.4877935552990422)

    def test_string_representation(self):
        assert str(self.index_var) == 'Refractive Index, Surface 1'


class TestAsphereCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        self.asphere_var = variable.AsphereCoeffVariable(self.optic, 1, 0)

    def test_get_value(self):
        assert np.isclose(self.asphere_var.get_value(), -2.248851)

    def test_update_value(self):
        self.asphere_var.update_value(-2.0)
        assert np.isclose(self.asphere_var.get_value(), -2.0)

    def test_get_value_no_scaling(self):
        self.optic = AsphericSinglet()
        self.asphere_var = variable.AsphereCoeffVariable(self.optic, 1, 0,
                                                         apply_scaling=False)
        assert np.isclose(self.asphere_var.get_value(), -0.0002248851)

    def test_string_representation(self):
        assert str(self.asphere_var) == 'Asphere Coeff. 0, Surface 1'


class TestPolynomialCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        poly_geo = PolynomialGeometry(CoordinateSystem(), 100,
                                      coefficients=np.zeros((3, 3)))
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.PolynomialCoeffVariable(self.optic, 0, (1, 1))

    def test_get_value(self):
        assert self.poly_var.get_value() == 0.0

    def test_update_value(self):
        self.poly_var.update_value(1.0)
        assert np.isclose(self.poly_var.get_value(), 1.0)

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
        assert np.isclose(self.poly_var.get_value(), 1.0)

    def test_string_representation(self):
        assert str(self.poly_var) == 'Poly. Coeff. (1, 1), Surface 0'

    def test_get_value_no_scaling(self):
        self.optic = AsphericSinglet()
        poly_geo = PolynomialGeometry(CoordinateSystem(), 100,
                                      coefficients=np.zeros((3, 3)))
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.PolynomialCoeffVariable(self.optic, 0, (1, 1),
                                                         apply_scaling=False)
        assert self.poly_var.get_value() == 0.0


class TestChebyshevCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        poly_geo = ChebyshevPolynomialGeometry(CoordinateSystem(), 100,
                                               coefficients=np.zeros((3, 3)))
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ChebyshevCoeffVariable(self.optic, 0, (1, 1))

    def test_get_value(self):
        assert self.poly_var.get_value() == 0.0

    def test_update_value(self):
        self.poly_var.update_value(1.0)
        assert np.isclose(self.poly_var.get_value(), 1.0)

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
        assert np.isclose(self.poly_var.get_value(), 1.0)

    def test_string_representation(self):
        assert str(self.poly_var) == 'Chebyshev Coeff. (1, 1), Surface 0'


class TestVariable:
    def test_get_value(self):
        optic = Objective60x()
        radius_var = variable.Variable(optic, 'radius', surface_number=1)
        assert np.isclose(radius_var.value, 4.5325999999999995)

    def test_unrecognized_attribute(self):
        optic = Objective60x()
        radius_var = variable.Variable(optic, 'radius', surface_number=1,
                                       unrecognized_attribute=1)
        assert np.isclose(radius_var.value, 4.5325999999999995)

    def test_invalid_type(self):
        optic = Objective60x()
        with pytest.raises(ValueError):
            variable.Variable(optic, 'invalid', surface_number=1)


class TestTiltVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.tilt_var_x = variable.TiltVariable(self.optic, 1, 'x')
        self.tilt_var_y = variable.TiltVariable(self.optic, 1, 'y')

    def test_get_value_x(self):
        assert np.isclose(self.tilt_var_x.get_value(), 0.0)

    def test_get_value_y(self):
        assert np.isclose(self.tilt_var_y.get_value(), 0.0)

    def test_update_value_x(self):
        self.tilt_var_x.update_value(5.0)
        assert np.isclose(self.tilt_var_x.get_value(), 5.0)

    def test_update_value_y(self):
        self.tilt_var_y.update_value(5.0)
        assert np.isclose(self.tilt_var_y.get_value(), 5.0)

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            variable.TiltVariable(self.optic, 1, 'z')

    def test_str(self):
        assert str(self.tilt_var_x) == 'Tilt X, Surface 1'
        assert str(self.tilt_var_y) == 'Tilt Y, Surface 1'

    def test_get_value_no_scaling(self):
        self.optic = Objective60x()
        self.tilt_var_x = variable.TiltVariable(self.optic, 1, 'x',
                                                apply_scaling=False)
        assert np.isclose(self.tilt_var_x.get_value(), 0.0)


class TestDecenterVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.decenter_var_x = variable.DecenterVariable(self.optic, 1, 'x')
        self.decenter_var_y = variable.DecenterVariable(self.optic, 1, 'y')

    def test_get_value_x(self):
        assert np.isclose(self.decenter_var_x.get_value(), 0.0)

    def test_get_value_y(self):
        assert np.isclose(self.decenter_var_y.get_value(), 0.0)

    def test_update_value_x(self):
        self.decenter_var_x.update_value(5.0)
        assert np.isclose(self.decenter_var_x.get_value(), 5.0)

    def test_update_value_y(self):
        self.decenter_var_y.update_value(5.0)
        assert np.isclose(self.decenter_var_y.get_value(), 5.0)

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            variable.DecenterVariable(self.optic, 1, 'z')

    def test_str(self):
        assert str(self.decenter_var_x) == 'Decenter X, Surface 1'
        assert str(self.decenter_var_y) == 'Decenter Y, Surface 1'

    def test_get_value_no_scaling(self):
        self.optic = Objective60x()
        self.decenter_var_x = variable.DecenterVariable(self.optic, 1, 'x',
                                                        apply_scaling=False)
        assert np.isclose(self.decenter_var_x.get_value(), 0.0)
